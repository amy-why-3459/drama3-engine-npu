"""Ascend grouped expert execution without per-expert Python dispatch."""

import torch
import torch_npu
from torch import nn


class NpuExperts(nn.Module):
    def __init__(self, reference: nn.Module):
        super().__init__()
        if reference.gate_up_proj.device.type != "npu":
            raise ValueError("Grouped expert execution requires NPU-resident weights")
        self.num_experts = reference.num_experts
        self.gate_up_proj = reference.gate_up_proj
        self.down_proj = reference.down_proj
        self.quantized = self.gate_up_proj.dtype == torch.int8
        if self.quantized:
            self.gate_up_proj = nn.Parameter(reference.gate_up_proj.transpose(1, 2).contiguous(), requires_grad=False)
            self.down_proj = nn.Parameter(reference.down_proj.transpose(1, 2).contiguous(), requires_grad=False)
            self.register_buffer('gate_up_proj_scale', reference.gate_up_proj_scale)
            self.register_buffer('down_proj_scale', reference.down_proj_scale)

    def matmul(self, inputs, weight, counts, scale=None):
        kwargs = {}
        if self.quantized:
            dtype = inputs.dtype
            inputs, token_scale = torch_npu.npu_dynamic_quant(inputs)
            kwargs = dict(scale=[scale], per_token_scale=[token_scale], output_dtype=dtype)
        return torch_npu.npu_grouped_matmul(
            x=[inputs], weight=[weight if self.quantized else weight.transpose(1, 2)],
            split_item=2, group_list_type=1, group_type=0, group_list=counts, **kwargs,
        )[0]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        if hidden_states.shape[0] == 0:
            return torch.empty_like(hidden_states)
        routed, inverse, counts, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            top_k_index.to(torch.int32),
            active_num=hidden_states.shape[0] * top_k_index.shape[1],
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=-1,
        )
        gate_up = self.matmul(routed, self.gate_up_proj, counts,
                              self.gate_up_proj_scale if self.quantized else None)
        activation = torch_npu.npu_swiglu(gate_up, dim=-1)
        projected = self.matmul(activation, self.down_proj, counts,
                                self.down_proj_scale if self.quantized else None)
        return torch_npu.npu_moe_token_unpermute(
            permuted_tokens=projected, sorted_indices=inverse,
            probs=top_k_weights.to(projected.dtype),
        )
