"""FP32-state GDN prefill via an optional Ascend kernel library.

This imports kernels only; neither SGLang's scheduler nor model runtime is used.
"""
import torch


def chunk_prefill(query, key, value, g, beta, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False,
                  cu_seqlens=None, **kwargs):
    from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu

    if kwargs:
        raise TypeError(f'Unsupported GDN arguments: {sorted(kwargs)}')
    batch, length, heads, dimension = query.shape
    if dimension != 128 or value.shape[-1] != 128 or cu_seqlens is not None:
        raise ValueError('Native GDN prefill requires dense batches and 128-dimensional K/V heads')
    if initial_state is None:
        initial_state = torch.zeros(batch, value.shape[-2], dimension, value.shape[-1],
                                    dtype=torch.float32, device=query.device)
    if initial_state.dtype != torch.float32:
        raise ValueError('Native GDN prefill requires FP32 recurrent state')
    cumulative = torch.arange(batch + 1, dtype=torch.int32, device=query.device) * length
    def flatten(tensor):
        return tensor.reshape(1, batch * length, *tensor.shape[2:]).contiguous()
    output, state, _ = chunk_gated_delta_rule_npu(
        q=flatten(query), k=flatten(key), v=flatten(value),
        g=flatten(g), beta=flatten(beta), initial_state=initial_state,
        output_final_state=output_final_state, cu_seqlens=cumulative,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return output.reshape(batch, length, value.shape[-2], value.shape[-1]), state
