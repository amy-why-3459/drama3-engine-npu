"""Check the existing NPU GDN prefill kernel against a recurrent CPU reference."""
import json
import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_recurrent_gated_delta_rule


def main():
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    torch.manual_seed(73)
    for length in (5, 65):
        q = torch.randn(1, length, 2, 128, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, length, 4, 128, dtype=torch.bfloat16)
        g = -torch.rand(1, length, 4, dtype=torch.float32)
        beta = torch.rand(1, length, 4, dtype=torch.bfloat16)
        initial = torch.randn(1, 4, 128, 128, dtype=torch.float32) * .1
        expected, state = torch_recurrent_gated_delta_rule(
            q.repeat_interleave(2, dim=2), k.repeat_interleave(2, dim=2), v,
            g, beta, initial.clone(), True, use_qk_l2norm_in_kernel=True,
        )
        output, final, _ = chunk_gated_delta_rule_npu(
            q.npu(), k.npu(), v.npu(), g.npu(), beta.npu(),
            initial_state=initial.npu(), output_final_state=True,
            cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device='npu'),
            use_qk_l2norm_in_kernel=True,
        )
        torch.npu.synchronize()
        output, final = output.cpu(), final.cpu()
        print(json.dumps({'length': length, 'output_max_abs': (output.float()-expected.float()).abs().max().item(),
                          'state_max_abs': (final-state).abs().max().item(),
                          'state_dtype': str(final.dtype)}), flush=True)
        torch.testing.assert_close(output, expected, atol=.02, rtol=.02)
        torch.testing.assert_close(final, state, atol=.02, rtol=.02)
        assert final.dtype == torch.float32
    print('GDN_PREFILL_SYNTHETIC_PARITY_PASS', flush=True)


if __name__ == '__main__':
    main()
