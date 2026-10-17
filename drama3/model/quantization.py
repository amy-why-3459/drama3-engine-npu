"""Explicit experimental quantization; the default checkpoint path stays BF16."""
import torch


def quantize_expert_weight(weight):
    if weight.ndim != 3 or not weight.is_floating_point():
        raise ValueError('Expected floating-point [experts, output, input] weights')
    # Process one expert at a time to bound temporary FP32 host memory.
    quantized = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(weight.shape[:-1], dtype=torch.bfloat16, device=weight.device)
    for expert in range(len(weight)):
        matrix = weight[expert].float()
        scale = matrix.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)/127
        quantized[expert] = (matrix/scale[:, None]).round().clamp(-127,127).to(torch.int8)
        scales[expert] = scale.to(torch.bfloat16)
    return quantized, scales
