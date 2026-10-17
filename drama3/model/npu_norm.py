"""Zero-centered RMSNorm, including reference BF16 residual-add rounding."""
import torch
import triton
import triton.language as tl
from torch import nn


@triton.jit
def _norm(X, W, R, Y, SUM, N: tl.constexpr, BLOCK: tl.constexpr,
          EPS: tl.constexpr, RESIDUAL: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + column, column < N, other=0).to(tl.float32)
    if RESIDUAL:
        residual = tl.load(R + row * N + column, column < N, other=0).to(tl.float32)
        x = (x + residual).to(X.dtype.element_ty).to(tl.float32)
        tl.store(SUM + row * N + column, x, column < N)
    gain = 1. + tl.load(W + column, column < N, other=0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    tl.store(Y + row * N + column, (x * inverse) * gain, column < N)


def gemma_rms_norm(x, weight, eps, residual=None):
    if x.device.type != 'npu' or weight.device != x.device or weight.numel() != x.shape[-1]:
        raise ValueError('Fused norm requires matching NPU tensors and last dimension')
    if residual is not None and (residual.shape != x.shape or residual.device != x.device or residual.dtype != x.dtype):
        raise ValueError('Residual must match the input shape, device and dtype')
    x = x.contiguous()
    result = torch.empty_like(x)
    summed = torch.empty_like(x) if residual is not None else x
    dimension = x.shape[-1]
    with torch.npu.device(x.device):
        _norm[(x.numel() // dimension,)](
            x, weight.contiguous(), residual.contiguous() if residual is not None else x,
            result, summed, N=dimension, BLOCK=triton.next_power_of_2(dimension),
            EPS=eps, RESIDUAL=residual is not None, num_warps=4, num_stages=1,
        )
    return result if residual is None else (result, summed)


class NpuGemmaRMSNorm(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.weight = reference.weight
        self.eps = reference.eps

    def forward(self, x, residual=None):
        return gemma_rms_norm(x, self.weight, self.eps, residual)
