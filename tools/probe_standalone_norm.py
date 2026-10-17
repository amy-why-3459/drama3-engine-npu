"""Check norm rounding and device placement independently of model generation."""
import json
import torch
import torch_npu
from drama3.model.npu_norm import gemma_rms_norm


def main():
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.manual_seed(734)
    results = []
    for dimension in (128, 256, 2048):
        for shape in ((1, dimension), (4, dimension), (2, 7, dimension)):
            for residual in (False, True):
                x = torch.randn(shape, dtype=torch.bfloat16)
                weight = torch.randn(dimension, dtype=torch.bfloat16) * .1
                r = torch.randn_like(x) if residual else None
                summed = x + r if residual else x
                h = summed.float()
                expected = (h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight.float())).to(x.dtype)
                actual = gemma_rms_norm(x.to('npu:1'), weight.to('npu:1'), 1e-6,
                                        r.to('npu:1') if residual else None)
                if residual:
                    actual, actual_sum = actual
                    torch.testing.assert_close(actual_sum.cpu(), summed, atol=0, rtol=0)
                torch.testing.assert_close(actual.cpu(), expected, atol=.015625, rtol=.005)
                results.append(dict(shape=shape, residual=residual, max_abs=float((actual.cpu()-expected).abs().max())))
    print(json.dumps(dict(status='PASS', cases=results), indent=2), flush=True)


if __name__ == '__main__':
    main()
