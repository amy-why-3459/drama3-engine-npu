"""Compare dense-batch NPU prefill and its state against CPU recurrence."""
import json
import torch
import torch_npu
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_recurrent_gated_delta_rule
from drama3.model.npu_gdn import chunk_prefill


def main():
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.manual_seed(732)
    results = []
    for batch, length in ((1, 5), (3, 65), (3, 1)):
        for initialized in (False, True):
            q = torch.randn(batch, length, 4, 128, dtype=torch.bfloat16)
            k, v = torch.randn_like(q), torch.randn_like(q)
            g = -torch.rand(batch, length, 4)
            beta = torch.rand(batch, length, 4, dtype=torch.bfloat16)
            initial = torch.randn(batch, 4, 128, 128) * .1 if initialized else None
            expected, state = torch_recurrent_gated_delta_rule(
                q, k, v, g, beta, initial, True, use_qk_l2norm_in_kernel=True)
            output, final = chunk_prefill(q.npu(), k.npu(), v.npu(), g.npu(), beta.npu(),
                                          initial_state=initial.npu() if initialized else None,
                                          output_final_state=True, use_qk_l2norm_in_kernel=True)
            output, final = output.cpu(), final.cpu()
            torch.testing.assert_close(output, expected, atol=.02, rtol=.02)
            torch.testing.assert_close(final, state, atol=.02, rtol=.02)
            assert final.dtype == torch.float32
            results.append(dict(batch=batch, length=length, initialized=initialized,
                                output_max_abs=float((output.float()-expected.float()).abs().max()),
                                state_max_abs=float((final-state).abs().max())))
    print(json.dumps(dict(status='PASS', cases=results), indent=2), flush=True)


if __name__ == '__main__':
    main()
