"""Single-step GDN recurrence with FP32 state and Transformers normalization."""
import torch
import triton
import triton.language as tl


@triton.jit
def _step(Q, K, V, G, BETA, S, OUT, NEXT, D: tl.constexpr, TILE: tl.constexpr):
    head = tl.program_id(0)
    column = tl.program_id(1) * TILE + tl.arange(0, TILE)
    row = tl.arange(0, D)
    q = tl.load(Q + head * D + row).to(tl.float32) * (D ** -.5)
    k = tl.load(K + head * D + row).to(tl.float32)
    v = tl.load(V + head * D + column).to(tl.float32)
    decay = tl.exp(tl.load(G + head).to(tl.float32))
    beta = tl.load(BETA + head).to(tl.float32)
    offset = head * D * D + row[:, None] * D + column[None, :]
    state = tl.load(S + offset) * decay
    delta = (v - tl.sum(state * k[:, None], axis=0)) * beta
    state = state + k[:, None] * delta[None, :]
    tl.store(NEXT + offset, state)
    tl.store(OUT + head * D + column, tl.sum(state * q[:, None], axis=0))


def recurrent_decode(query, key, value, g, beta, initial_state,
                     output_final_state, use_qk_l2norm_in_kernel=False):
    if (query.shape[1] != 1 or query.shape[-1] != 128
            or query.shape != key.shape or query.shape != value.shape
            or initial_state is None or initial_state.dtype != torch.float32):
        raise ValueError('Fused decode requires one token, equal 128-dimensional heads and FP32 state')
    expected_state = (query.shape[0], query.shape[2], 128, 128)
    if initial_state.shape != expected_state or g.shape != query.shape[:3] or beta.shape != g.shape:
        raise ValueError('Incompatible recurrent state or gate shape')
    if any(t.device != query.device for t in (key, value, g, beta, initial_state)):
        raise ValueError('Recurrent tensors must share a device')
    if use_qk_l2norm_in_kernel:
        # Retain the reference BF16 intermediate rounding, not an FP32 substitute.
        query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    batch, _, heads, dimension = query.shape
    output = torch.empty_like(value)
    final = torch.empty_like(initial_state)
    # Triton launches on the current device, unlike dispatcher-managed ATen ops.
    # Whole-layer placement may leave the caller on a different NPU.
    with torch.npu.device(query.device):
        _step[(batch * heads, 4)](
            query.contiguous(), key.contiguous(), value.contiguous(),
            g.contiguous(), beta.contiguous(), initial_state.contiguous(), output, final,
            D=dimension, TILE=32, num_warps=4, num_stages=1,
        )
    return output, final if output_final_state else None
