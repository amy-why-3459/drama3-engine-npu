"""Native Ascend decode attention over the standalone tower's slot cache."""
import torch
import torch_npu


def paged_decode(query, keys, values, block_table, context_lens_cpu):
    """Read 16-token pages directly, without gathering each request's KV history."""
    if query.shape[-1] not in (128, 256) or keys.shape[1] % 16:
        raise ValueError('Validated native tower attention requires head 128/256 and 16-token pages')
    if context_lens_cpu is None or context_lens_cpu.device.type != 'cpu':
        raise ValueError('Explicit CPU context lengths are required')
    heads, dimension = query.shape[-2:]
    kv_heads = keys.shape[-2]
    output = torch.empty_like(query)
    torch_npu._npu_paged_attention(
        query=query.contiguous(),
        key_cache=keys.view(-1, 16, kv_heads, dimension),
        value_cache=values.view(-1, 16, kv_heads, dimension),
        num_heads=heads, num_kv_heads=kv_heads, scale_value=dimension ** -.5,
        block_table=block_table, context_lens=context_lens_cpu, out=output,
    )
    return output
