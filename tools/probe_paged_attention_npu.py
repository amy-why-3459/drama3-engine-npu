"""Check native paged attention layouts against explicit softmax attention."""

import json

import torch
import torch_npu


torch.set_num_threads(4)
torch.npu.set_device(0)
torch.manual_seed(18)
for head_dim in (128, 256):
    for page_size in (1, 16, 128):
        query = torch.randn(1, 4, head_dim, dtype=torch.bfloat16, device='npu')
        keys = torch.randn(4, page_size, 2, head_dim, dtype=torch.bfloat16, device='npu')
        values = torch.randn_like(keys)
        blocks = torch.tensor([[1, 2, 3]], dtype=torch.int32, device='npu')
        length = 3
        if page_size == 1:
            flat_k, flat_v = keys[1:4, 0], values[1:4, 0]
        else:
            flat_k, flat_v = keys[1, :length], values[1, :length]
        k = flat_k.float().transpose(0, 1).repeat_interleave(2, 0)
        v = flat_v.float().transpose(0, 1).repeat_interleave(2, 0)
        expected = ((query.float()[0, :, None] @ k.transpose(-2, -1))
                    * head_dim ** -.5).softmax(-1) @ v
        output = torch.full_like(query, float('nan'))
        try:
            returned = torch_npu._npu_paged_attention(
                query=query, key_cache=keys, value_cache=values, num_heads=4,
                num_kv_heads=2, scale_value=head_dim ** -.5, block_table=blocks,
                context_lens=torch.tensor([length], dtype=torch.int32), out=output,
            )
            torch.npu.synchronize()
            row = {'head_dim': head_dim, 'page_size': page_size,
                   'return_type': type(returned).__name__,
                   'finite': bool(torch.isfinite(output).all()),
                   'max_abs': (output.float()[0] - expected[:, 0]).abs().max().item()}
        except Exception as error:
            row = {'head_dim': head_dim, 'page_size': page_size, 'error': str(error)}
        print(json.dumps(row), flush=True)
        if page_size in (16, 128):
            if 'error' in row:
                raise RuntimeError(row['error'])
            torch.testing.assert_close(output.float()[0], expected[:, 0], atol=.02, rtol=.02)
print('NATIVE_PAGE16_PAGE128_PARITY_PASS', flush=True)
