"""Explore dynamic W8A8 MoE; quantization error is reported, not assumed acceptable."""
import json
import time
import torch
import torch_npu
from torch import nn
from drama3.model.npu_experts import NpuExperts


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(24)
    torch.npu.set_device(0)
    reference = nn.Module()
    reference.num_experts = 256
    reference.gate_up_proj = nn.Parameter(torch.randn(256, 1024, 2048, device='npu:0', dtype=torch.bfloat16)*.03, requires_grad=False)
    reference.down_proj = nn.Parameter(torch.randn(256, 2048, 512, device='npu:0', dtype=torch.bfloat16)*.03, requires_grad=False)
    eager = NpuExperts(reference)
    def pack(weight):
        scale = weight.float().abs().amax(-1).clamp_min(1e-8)/127
        quantized = (weight.float()/scale[..., None]).round().clamp(-127, 127).to(torch.int8)
        return quantized.transpose(1, 2).contiguous(), scale.to(torch.bfloat16)
    up, us = pack(reference.gate_up_proj)
    down, ds = pack(reference.down_proj)
    def candidate(hidden, ids, probs):
        routed, inverse, counts, _ = torch_npu.npu_moe_init_routing_v2(hidden, ids.to(torch.int32),
            active_num=hidden.shape[0]*ids.shape[1], expert_num=256, expert_tokens_num_type=1,
            expert_tokens_num_flag=True, active_expert_range=[0,256], quant_mode=-1)
        def mm(x, weight, scale):
            q, token_scale = torch_npu.npu_dynamic_quant(x)
            return torch_npu.npu_grouped_matmul(x=[q], weight=[weight], scale=[scale],
                per_token_scale=[token_scale], split_item=2, group_list_type=1,
                group_type=0, group_list=counts, output_dtype=torch.bfloat16)[0]
        output = mm(torch_npu.npu_swiglu(mm(routed, up, us)), down, ds)
        return torch_npu.npu_moe_token_unpermute(output, inverse, probs=probs.to(output.dtype))
    results = []
    for batch in (1, 8, 64):
        hidden = torch.randn(batch, 2048, device='npu:0', dtype=torch.bfloat16)
        probs, ids = torch.randn(batch, 256, device='npu:0').softmax(-1).topk(8)
        probs /= probs.sum(-1, keepdim=True)
        expected, actual = eager(hidden, ids, probs), candidate(hidden, ids, probs)
        assert torch.isfinite(actual).all()
        record = dict(batch=batch, max_abs=float((expected-actual).abs().max()),
            relative_l2=float((expected.float()-actual.float()).norm()/expected.float().norm()),
            rounds=[])
        for _ in range(5): eager(hidden, ids, probs); candidate(hidden, ids, probs)
        for name, function in (('bf16', eager), ('int8', candidate), ('int8', candidate), ('bf16', eager)):
            torch.npu.synchronize()
            start=time.perf_counter()
            for _ in range(30): function(hidden, ids, probs)
            torch.npu.synchronize()
            record['rounds'].append(dict(arm=name, ms=(time.perf_counter()-start)*1000/30))
        results.append(record)
    print(json.dumps(dict(status='EXPLORATORY_ONLY', quality='NOT_VALIDATED', results=results)), flush=True)


if __name__ == '__main__': main()
