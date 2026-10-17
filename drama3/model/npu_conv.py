"""Single-token causal depthwise convolution without layout conversions."""
import torch
import triton
import triton.language as tl


@triton.jit
def _update(X, STATE, W, BIAS, Y, D: tl.constexpr, K: tl.constexpr,
            HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    tap = tl.arange(0, K)
    valid = row < ROWS
    state = tl.load(STATE + row[:, None]*K + tap[None, :], valid[:, None], other=0)
    next_tap = tl.broadcast_to(((tap+1) % K)[None, :], (BLOCK, K))
    shifted = tl.gather(state, next_tap, 1)
    current = tl.load(X + row, valid, other=0)
    shifted = tl.where(tap[None, :] == K-1, current[:, None], shifted)
    weight = tl.load(W + (row % D)[:, None]*K + tap[None, :], valid[:, None], other=0)
    total = tl.sum(shifted.to(tl.float32)*weight.to(tl.float32), axis=1)
    tl.store(STATE + row[:, None]*K + tap[None, :], shifted, valid[:, None])
    if HAS_BIAS:
        total += tl.load(BIAS + row % D, valid, other=0).to(tl.float32)
    total = total.to(X.dtype.element_ty).to(tl.float32)
    output = total / (1. + tl.exp(-total))
    tl.store(Y + row, output, valid)


def causal_conv_update(hidden_states, conv_state, weight, bias=None, activation=None):
    if (hidden_states.device.type != 'npu' or hidden_states.ndim != 3
            or hidden_states.shape[-1] != 1 or activation not in (None, 'silu', 'swish')):
        raise ValueError('Fused convolution requires single-token NPU SiLU inputs')
    batch, width, _ = hidden_states.shape
    if batch > 8:
        # Measured native convolution is faster for large batches on 910B1.
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_causal_conv1d_update

        return torch_causal_conv1d_update(hidden_states, conv_state, weight, bias, activation)
    if (conv_state.shape[:2] != (batch, width) or weight.shape != conv_state.shape[1:]
            or not conv_state.is_contiguous() or conv_state.device != hidden_states.device
            or weight.device != hidden_states.device or conv_state.dtype != hidden_states.dtype
            or weight.dtype != hidden_states.dtype):
        raise ValueError('Convolution weights and contiguous state must match inputs')
    if weight.shape[-1] != 4:
        raise ValueError('Fused convolution currently requires four taps')
    x = hidden_states.contiguous()
    output = torch.empty_like(x)
    with torch.npu.device(x.device):
        _update[(triton.cdiv(batch*width, 256),)](
            x, conv_state, weight.contiguous(), bias if bias is not None else x, output,
            D=width, K=weight.shape[-1], HAS_BIAS=bias is not None, BLOCK=256, ROWS=batch*width,
            num_warps=4, num_stages=1, enable_fp_fusion=False,
        )
    return output
