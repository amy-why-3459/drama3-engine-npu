"""Device-local graphs for row-independent tower work; dynamic attention stays eager."""
import torch
import torch_npu  # noqa: F401
import triton
from torch import nn
from drama3.model.tower import interleaved_rope


class RowGraph:
    def __init__(self, function, tensors, bucket):
        self.inputs = [x.new_zeros((bucket, *x.shape[1:])) for x in tensors]
        self.function = function
        self.copy(tensors)
        for _ in range(3):
            function(*self.inputs)
        torch.npu.synchronize(tensors[0].device)
        self.stream = torch.npu.Stream(device=tensors[0].device)
        self.graph = torch.npu.NPUGraph()
        with torch.npu.graph(self.graph, stream=self.stream):
            self.output = function(*self.inputs)

    def copy(self, tensors):
        for target, source in zip(self.inputs, tensors):
            if len(source) == len(target):
                target.copy_(source)
                continue
            target[:len(source)].copy_(source)
            if len(source) < len(target):
                target[len(source):].zero_()

    def replay(self, tensors):
        self.copy(tensors)
        self.graph.replay()
        rows = len(tensors[0])
        if rows == len(self.inputs[0]):
            return self.output
        if isinstance(self.output, tuple):
            return tuple(value[:rows] for value in self.output)
        return self.output[:rows]


class GraphTowerLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.device = next(module.parameters()).device
        self.entries = {}

    def before_attention(self, hidden, code_ids, valid, extra, positions):
        m = self.module
        table = getattr(m, m.embedding_name)
        embedded = torch.where(valid[:, None], table(code_ids.clamp(0, table.num_embeddings - 1)), 0.)
        embedded = embedded + extra.to(embedded.dtype)
        mixed = m.eh_proj(torch.cat((m.enorm(embedded), m.hnorm(hidden)), dim=-1))
        block = m.transformer_layer
        attention = block.attention
        rows = len(hidden)
        q_width = attention.n_head * attention.head_dim
        kv_width = attention.n_kv * attention.head_dim
        query, gate, key, value = attention.wqkv(block.attention_norm(mixed)).split(
            [q_width, q_width, kv_width, kv_width], dim=-1)
        query = attention.q_norm(query.reshape(-1, attention.head_dim)).view(
            rows, attention.n_head, attention.head_dim)
        key = attention.k_norm(key.reshape(-1, attention.head_dim)).view(
            rows, attention.n_kv, attention.head_dim)
        query = interleaved_rope(query, positions, attention.rotary_dim, attention.rope_base)
        key = interleaved_rope(key, positions, attention.rotary_dim, attention.rope_base)
        return mixed, query, gate, key, value.reshape(rows, attention.n_kv, attention.head_dim)

    def after_attention(self, attended, gate, mixed):
        m = self.module
        attention = m.transformer_layer.attention
        attended = attention.wo(attended.flatten(1) * torch.sigmoid(gate))
        value, residual = m.transformer_layer.ffn_norm(attended, mixed)
        out, _ = m.final_layernorm(m.transformer_layer.feed_forward(value), residual)
        return out

    def run_graph(self, stage, tensors):
        key = (stage, triton.next_power_of_2(len(tensors[0])))
        with torch.npu.device(self.device):
            if key not in self.entries:
                function = self.before_attention if stage == 'before' else self.after_attention
                self.entries[key] = RowGraph(function, tensors, key[1])
            return self.entries[key].replay(tensors)

    def forward(self, hidden, code_ids, code_is_valid, extra_input, batch, key_cache, value_cache):
        if batch.max_rows != 1:
            return self.module(hidden, code_ids, code_is_valid, extra_input, batch, key_cache, value_cache)
        extra = torch.zeros_like(hidden) if extra_input is None else extra_input
        mixed, query, gate, key, value = self.run_graph(
            'before', (hidden, code_ids, code_is_valid, extra, batch.positions))
        # Only live rows write the request-owned cache. Bucket padding must not
        # write a shared scratch address or change a request's history.
        key_cache[batch.slots, batch.positions] = key.to(key_cache.dtype)
        value_cache[batch.slots, batch.positions] = value.to(value_cache.dtype)
        attention = self.module.transformer_layer.attention
        if attention.use_npu_paged_attention and batch.npu_block_table is not None:
            from drama3.model.npu_attention import paged_decode
            attended = paged_decode(query, key_cache, value_cache,
                                    batch.npu_block_table, batch.seq_used_cpu)
        else:
            attended = attention._attend_reference(query, batch, key_cache, value_cache)
        return self.run_graph('after', (attended, gate, mixed))

    def clear_graphs(self):
        with torch.npu.device(self.device):
            torch.npu.synchronize(self.device)
            # Destroy each graph and its stream while its device is current.
            # A for-loop variable would retain the last entry past this guard.
            while self.entries:
                _, entry = self.entries.popitem()
                entry.graph.reset()
                del entry
