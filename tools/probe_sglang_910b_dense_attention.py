"""Diagnose decode attention using exact pooled KV and a Torch math reference.

This slow diagnostic is not an optimized baseline or a performance candidate.
"""
import runpy
import os
import json
from pathlib import Path

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

original_decode = AscendAttnBackend.forward_decode


def decode(self, q, k, v, layer, forward_batch, save_kv_cache=True,
           q_rope=None, k_rope=None, topk_indices=None, sinks=None, slopes=None):
    assert self.page_size in (1, 16, 128) and not layer.is_cross_attention
    assert q_rope is None and k_rope is None and sinks is None and slopes is None
    assert layer.sliding_window_size == -1
    if save_kv_cache:
        self.token_to_kv_pool.set_kv_buffer(layer, KVWriteLoc(forward_batch.out_cache_loc), k, v)
    key = self.token_to_kv_pool.get_key_buffer(layer.layer_id).view(-1, layer.tp_k_head_num, layer.qk_head_dim)
    value = self.token_to_kv_pool.get_value_buffer(layer.layer_id).view(-1, layer.tp_v_head_num, layer.v_head_dim)
    queries = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    output = []
    for row, (request_index, length) in enumerate(zip(forward_batch.req_pool_indices.cpu().tolist(), forward_batch.seq_lens.cpu().tolist())):
        slots = self.req_to_token[request_index, :length].long()
        keys = key[slots].transpose(0, 1).unsqueeze(0).repeat_interleave(layer.tp_q_head_num // layer.tp_k_head_num, dim=1)
        values = value[slots].transpose(0, 1).unsqueeze(0).repeat_interleave(layer.tp_q_head_num // layer.tp_v_head_num, dim=1)
        assert torch.isfinite(keys).all() and torch.isfinite(values).all()
        with sdpa_kernel(SDPBackend.MATH):
            attended = torch.nn.functional.scaled_dot_product_attention(
                queries[row].unsqueeze(0).unsqueeze(2), keys, values, scale=layer.scaling,
            )
        output.append(attended.reshape(-1))
    reference = torch.stack(output)
    if os.environ.get('DRAMA3_COMPARE_NATIVE_ATTENTION') == '1':
        native = original_decode(self, q, k, v, layer, forward_batch, save_kv_cache=False)
        contiguous = original_decode(self, q.contiguous(), k, v, layer, forward_batch,
                                     save_kv_cache=False)
        slots_match = all(torch.equal(
            self.forward_metadata.block_tables[row, :(length + self.page_size - 1) // self.page_size].long(),
            self.req_to_token[index, :length:self.page_size].long() // self.page_size,
        ) for row, (index, length) in enumerate(zip(
            forward_batch.req_pool_indices.cpu().tolist(), forward_batch.seq_lens.cpu().tolist())))
        print('NATIVE_ATTENTION_COMPARISON ' + json.dumps({
            'layer': layer.layer_id, 'slots_match': slots_match,
            'metadata_lengths': self.forward_metadata.seq_lens_cpu_int.tolist(),
            'request_lengths': forward_batch.seq_lens.cpu().tolist(),
            'native_finite': bool(torch.isfinite(native).all()),
            'contiguous_finite': bool(torch.isfinite(contiguous).all()),
            'contiguous_max_abs': (contiguous.float() - reference.float()).abs().max().item(),
            'native_abs_max': native.float().abs().max().item(),
            'reference_abs_max': reference.float().abs().max().item(),
            'max_abs': (native.float() - reference.float()).abs().max().item(),
            'q_shape': list(q.shape), 'q_stride': list(q.stride()),
            'key_shape': list(key.shape),
        }), flush=True)
    return reference


AscendAttnBackend.forward_decode = decode
runpy.run_path(str(Path(__file__).with_name('probe_sglang_910b_prefill.py')), run_name='__main__')
