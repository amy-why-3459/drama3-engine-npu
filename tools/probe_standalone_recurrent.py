"""Measure and validate FP32-state one-step recurrence on Ascend."""
import argparse
import json
import time
import torch
import torch_npu
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_recurrent_gated_delta_rule
from drama3.model.npu_recurrent import recurrent_decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensor-device', type=int, default=0)
    args = parser.parse_args()
    device = f'npu:{args.tensor_device}'
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.manual_seed(733)
    results = []
    for batch in (1, 8, 32):
        q = torch.randn(batch, 1, 32, 128, dtype=torch.bfloat16, device=device)
        k, v = torch.randn_like(q), torch.randn_like(q)
        g = -torch.rand(batch, 1, 32, device=device)
        beta = torch.rand(batch, 1, 32, dtype=torch.bfloat16, device=device)
        initial = torch.randn(batch, 32, 128, 128, device=device) * .1
        expected, state = torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial, True, True)
        output, final = recurrent_decode(q, k, v, g, beta, initial, True, True)
        torch.testing.assert_close(output.cpu(), expected.cpu(), atol=.002, rtol=.002)
        torch.testing.assert_close(final.cpu(), state.cpu(), atol=2e-5, rtol=2e-5)
        reference_state, fused_state = initial, initial
        for _ in range(32):
            expected, reference_state = torch_recurrent_gated_delta_rule(q, k, v, g, beta, reference_state, True, True)
            output, fused_state = recurrent_decode(q, k, v, g, beta, fused_state, True, True)
        torch.testing.assert_close(output.cpu(), expected.cpu(), atol=.002, rtol=.002)
        torch.testing.assert_close(fused_state.cpu(), reference_state.cpu(), atol=2e-5, rtol=2e-5)
        timing = {}
        for name, function in [('reference', torch_recurrent_gated_delta_rule), ('fused', recurrent_decode)]:
            for _ in range(5):
                function(q, k, v, g, beta, initial, True, True)
            torch.npu.synchronize(device)
            start = time.perf_counter()
            for _ in range(50):
                function(q, k, v, g, beta, initial, True, True)
            torch.npu.synchronize(device)
            timing[name + '_ms'] = (time.perf_counter() - start) * 1000 / 50
        results.append(dict(batch=batch, output_max_abs=float((output-expected).abs().max()),
                            state_max_abs=float((final-state).abs().max()), **timing))
    print(json.dumps(dict(status='PASS', scope='component_only_single_round', cases=results), indent=2), flush=True)


if __name__ == '__main__':
    main()
