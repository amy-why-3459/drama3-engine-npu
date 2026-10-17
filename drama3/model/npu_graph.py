"""Bucketed NPU graph replay of independent cached GDN rows.

Only single-token decode is captured. Prefill uses the original implementation;
request caches are copied into and out of graph-owned buffers on each replay.
"""
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
import triton
from torch import nn


class StaticGDNCache:
    def __init__(self, layer_index, conv, recurrent):
        self.layers = {layer_index: SimpleNamespace(conv_states={0: conv}, recurrent_states={0: recurrent})}

    def has_previous_state(self, layer_index):
        return layer_index in self.layers

    def update_conv_state(self, state, layer_index):
        self.layers[layer_index].conv_states[0].copy_(state)

    def update_recurrent_state(self, state, layer_index):
        self.layers[layer_index].recurrent_states[0].copy_(state)


class GraphEntry:
    def __init__(self, module, inputs, source, bucket, *, decoder=False):
        self.inputs = inputs.new_zeros((bucket, *inputs.shape[1:]))
        self.conv = source.conv_states[0].new_zeros((bucket, *source.conv_states[0].shape[1:]))
        self.recurrent = source.recurrent_states[0].new_zeros((bucket, *source.recurrent_states[0].shape[1:]))
        self.module = module
        self.decoder = decoder
        index = module.linear_attn.layer_idx if decoder else module.layer_idx
        self.cache = StaticGDNCache(index, self.conv, self.recurrent)
        self.copy_inputs(inputs, source)
        for _ in range(3):
            self.invoke()
        torch.npu.synchronize(inputs.device)
        self.graph = torch.npu.NPUGraph()
        self.capture_stream = torch.npu.Stream(device=inputs.device)
        with torch.npu.graph(self.graph, stream=self.capture_stream):
            self.output = self.invoke()

    def invoke(self):
        if self.decoder:
            return self.module(self.inputs, position_embeddings=None, past_key_values=self.cache, attention_mask=None)
        return self.module(self.inputs, cache_params=self.cache, attention_mask=None)

    def copy_inputs(self, inputs, source):
        batch = len(inputs)
        self.inputs[:batch].copy_(inputs)
        self.conv[:batch].copy_(source.conv_states[0])
        self.recurrent[:batch].copy_(source.recurrent_states[0])
        if batch < len(self.inputs):
            self.inputs[batch:].zero_()
            self.conv[batch:].zero_()
            self.recurrent[batch:].zero_()

    def replay(self, inputs, source):
        self.copy_inputs(inputs, source)
        self.graph.replay()
        batch = len(inputs)
        source.conv_states[0].copy_(self.conv[:batch])
        source.recurrent_states[0].copy_(self.recurrent[:batch])
        return self.output[:batch]


class GraphGDN(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.entries = {}
        self.device = next(module.parameters()).device

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        if (hidden_states.shape[1] != 1 or cache_params is None or attention_mask is not None
                or not cache_params.has_previous_state(self.module.layer_idx)
                or set(kwargs) - {'use_cache'} or kwargs.get('use_cache', True) is False):
            return self.module(hidden_states, cache_params=cache_params, attention_mask=attention_mask, **kwargs)
        source = cache_params.layers[self.module.layer_idx]
        if source.number_of_states != 1 or source.record_past:
            raise ValueError('GDN graphs require a single current state per request')
        bucket = triton.next_power_of_2(len(hidden_states))
        with torch.npu.device(self.device):
            if bucket not in self.entries:
                self.entries[bucket] = GraphEntry(self.module, hidden_states, source, bucket)
            return self.entries[bucket].replay(hidden_states, source)

    def clear_graphs(self):
        with torch.npu.device(self.device):
            torch.npu.synchronize(self.device)
            for entry in self.entries.values():
                entry.graph.reset()
            self.entries.clear()


class GraphGDNDecoder(GraphGDN):
    """Capture the GDN block, norms and grouped MoE together."""
    def __init__(self, module):
        super().__init__(module)
        self.block_type = module.block_type
        self.layer_index = module.linear_attn.layer_idx

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                position_ids=None, past_key_values=None, **kwargs):
        if (hidden_states.shape[1] != 1 or past_key_values is None or attention_mask is not None
                or not past_key_values.has_previous_state(self.layer_index)
                or set(kwargs) - {'use_cache'} or kwargs.get('use_cache', True) is False):
            return self.module(hidden_states, position_embeddings=position_embeddings,
                               attention_mask=attention_mask, position_ids=position_ids,
                               past_key_values=past_key_values, **kwargs)
        source = past_key_values.layers[self.layer_index]
        if source.number_of_states != 1 or source.record_past:
            raise ValueError('Decoder graphs require a single current state per request')
        hidden_states = hidden_states.to(self.device)
        bucket = triton.next_power_of_2(len(hidden_states))
        with torch.npu.device(self.device):
            if bucket not in self.entries:
                self.entries[bucket] = GraphEntry(self.module, hidden_states, source, bucket, decoder=True)
            return self.entries[bucket].replay(hidden_states, source)
