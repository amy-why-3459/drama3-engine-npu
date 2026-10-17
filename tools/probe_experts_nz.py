"""Synthetic correctness and latency probe; not an end-to-end speedup claim."""

import argparse
import json
import time

import torch
import torch_npu  # noqa: F401
from torch import nn
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts
from transformers import Qwen3_5MoeTextConfig

from drama3.model.npu_experts import NpuExperts


torch.set_num_threads(4)
torch.npu.set_device(0)
torch.manual_seed(24)
parser = argparse.ArgumentParser()
parser.add_argument("--real-shape", action="store_true")
parser.add_argument("--nz", action="store_true")
args = parser.parse_args()
expert_count, hidden_dim, intermediate_dim, top_k = (
    (256, 2048, 512, 8) if args.real_shape else (8, 128, 128, 2)
)
with torch.device("meta"):
    experts = Qwen3_5MoeExperts(Qwen3_5MoeTextConfig(
        num_experts=expert_count, hidden_size=hidden_dim,
        moe_intermediate_size=intermediate_dim, hidden_act="silu",
    ))
experts.gate_up_proj = nn.Parameter(torch.randn(expert_count, 2 * intermediate_dim, hidden_dim, device="npu", dtype=torch.bfloat16) * 0.03, requires_grad=False)
experts.down_proj = nn.Parameter(torch.randn(expert_count, hidden_dim, intermediate_dim, device="npu", dtype=torch.bfloat16) * 0.03, requires_grad=False)
grouped = NpuExperts(experts)
if args.nz:
    torch.npu.config.allow_internal_format = True
    grouped.gate_up_proj = nn.Parameter(torch_npu.npu_format_cast(experts.gate_up_proj, 29), requires_grad=False)
    grouped.down_proj = nn.Parameter(torch_npu.npu_format_cast(experts.down_proj, 29), requires_grad=False)
    print('FORMATS', torch_npu.get_npu_format(grouped.gate_up_proj), torch_npu.get_npu_format(grouped.down_proj), flush=True)
records = []
with torch.inference_mode():
    for batch in (1, 2, 8, 32, 64):
        hidden = torch.randn(batch, hidden_dim, device="npu", dtype=torch.bfloat16)
        weights, indices = torch.randn(batch, expert_count, device="npu").softmax(-1).topk(top_k)
        weights /= weights.sum(-1, keepdim=True)
        expected = experts(hidden, indices, weights)
        actual = grouped(hidden, indices, weights)
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.04)
        record = {"batch": batch, "max_abs_error": (actual - expected).abs().max().item()}
        for name, function in (("reference", lambda: experts(hidden, indices, weights)), ("grouped", lambda: grouped(hidden, indices, weights))):
            for _ in range(3):
                function()
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(20):
                function()
            torch.npu.synchronize()
            record[name + "_ms"] = (time.perf_counter() - started) * 1000 / 20
        records.append(record)
print(json.dumps(records, indent=2), flush=True)
print("SYNTHETIC_GROUPED_EXPERTS_PASS", flush=True)
