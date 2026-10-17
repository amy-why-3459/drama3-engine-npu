"""Framework-independent PyTorch DRAMA-3 backbone and codebook tower loading."""

from __future__ import annotations

from functools import partial
import math
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeTextRotaryEmbedding,
    Qwen3_5MoeRMSNorm,
    torch_causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from drama3.model.checkpoint import Checkpoint
from drama3.model.configuration_fish_omni import (
    FishOmniAudioDecoderConfig,
    to_qwen3_5_moe_text_config,
)
from drama3.model.tower import CodebookTower, GemmaRMSNorm, TowerAttention


def move_tensors(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_tensors(element, device) for element in value)
    if isinstance(value, dict):
        return {key: move_tensors(element, device) for key, element in value.items()}
    return value


def move_layer_inputs(module, inputs, keywords, *, device):
    return move_tensors(inputs, device), move_tensors(keywords, device)


class Drama3Model(nn.Module):
    """Reference eager implementation with whole-layer placement across devices."""

    def __init__(self, config: dict, cache_slots: int, max_length: int):
        super().__init__()
        self.config = config
        text_config = to_qwen3_5_moe_text_config(config["text_config"])
        text_config._attn_implementation = "sdpa"
        self.backbone = Qwen3_5MoeForCausalLM(text_config)
        decoder_config = FishOmniAudioDecoderConfig(**config["audio_decoder_config"])
        self.tower = CodebookTower(
            decoder_config,
            semantic_begin_id=config["semantic_start_token_id"],
            cache_slots=cache_slots,
            max_len=max_length,
        )
        self.devices: tuple[torch.device, ...] = ()
        self.layer_hooks = []

    @classmethod
    def load(
        cls, directory: str | Path, devices: list[str], cache_slots=1, max_length=1024,
        use_npu_moe=False, use_npu_attention=False, use_npu_gdn_prefill=False, use_npu_gdn_decode=False, use_npu_norm=False, use_npu_gdn_graph=False, use_npu_layer_graph=False,
    ):
        if not devices or cache_slots < 1 or max_length < 1:
            raise ValueError("Nonempty devices and positive cache bounds are required")
        if use_npu_layer_graph and (use_npu_gdn_graph or not use_npu_moe):
            raise ValueError('Decoder layer graphs require grouped NPU MoE and exclude nested GDN graphs')
        checkpoint = Checkpoint(directory)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                model = cls(checkpoint.config, cache_slots, max_length)
        finally:
            torch.set_default_dtype(previous_dtype)
        model.devices = tuple(torch.device(device) for device in devices)
        layer_count = len(model.backbone.model.layers)
        layer_devices = [
            model.devices[min(index * len(devices) // layer_count, len(devices) - 1)]
            for index in range(layer_count)
        ]
        try:
            for name, parameter in list(model.named_parameters()):
                if name.startswith("tower."):
                    source_name = name.removeprefix("tower.")
                    target_device = model.devices[-1]
                    value = checkpoint.tensor(source_name, parameter.dtype)
                else:
                    local_name = name.removeprefix("backbone.")
                    target_device = model.devices[-1]
                    if local_name.startswith("model.layers."):
                        layer_index = int(local_name.split(".")[2])
                        target_device = layer_devices[layer_index]
                    elif local_name.startswith("model.embed_tokens."):
                        target_device = model.devices[0]
                    source_name = (
                        local_name.replace("model.", "model.language_model.", 1)
                        if local_name.startswith("model.")
                        else local_name
                    )
                    if local_name.endswith(
                        (".experts.gate_up_proj", ".experts.down_proj")
                    ):
                        value = torch.empty(parameter.shape, dtype=parameter.dtype)
                        prefix, kind = source_name.rsplit(".", 1)
                        intermediate = checkpoint.config["text_config"][
                            "moe_intermediate_size"
                        ]
                        for expert in range(parameter.shape[0]):
                            if kind == "gate_up_proj":
                                value[expert, :intermediate] = checkpoint.tensor(
                                    f"{prefix}.{expert}.gate_proj.weight",
                                    parameter.dtype,
                                )
                                value[expert, intermediate:] = checkpoint.tensor(
                                    f"{prefix}.{expert}.up_proj.weight", parameter.dtype
                                )
                            else:
                                value[expert] = checkpoint.tensor(
                                    f"{prefix}.{expert}.down_proj.weight",
                                    parameter.dtype,
                                )
                    else:
                        value = checkpoint.tensor(source_name, parameter.dtype)
                if value.shape != parameter.shape:
                    raise ValueError(
                        f"{name}: checkpoint {tuple(value.shape)} != model {tuple(parameter.shape)}"
                    )
                parent_name, leaf = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                setattr(
                    parent,
                    leaf,
                    nn.Parameter(value.to(target_device), requires_grad=False),
                )
                if ".experts.down_proj" in name:
                    print(f"Loaded {name} on {target_device}", flush=True)
            checkpoint.check_complete()
        finally:
            checkpoint.close()
        model.backbone.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(
            model.backbone.config, device=model.devices[0]
        )
        model.tower.row_offsets = (
            torch.arange(model.tower.num_codebooks, device=model.devices[-1])
            * model.tower.codebook_size
        )
        for layer, device in zip(model.backbone.model.layers, layer_devices):
            if use_npu_moe and device.type == "npu":
                from drama3.model.npu_experts import NpuExperts

                if model.backbone.config.hidden_act != "silu":
                    raise ValueError("NPU grouped MoE currently supports SiLU only")
                layer.mlp.experts = NpuExperts(layer.mlp.experts)
            model.layer_hooks.append(
                layer.register_forward_pre_hook(
                    partial(move_layer_inputs, device=device), with_kwargs=True
                )
            )
            if hasattr(layer, "linear_attn") and device.type != "cuda":
                linear = layer.linear_attn
                linear.causal_conv1d_fn = None
                linear.causal_conv1d_update = torch_causal_conv1d_update
                linear.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                linear.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
                if use_npu_gdn_prefill and device.type == "npu":
                    from drama3.model.npu_gdn import chunk_prefill

                    linear.chunk_gated_delta_rule = chunk_prefill
                if use_npu_gdn_decode and device.type == "npu":
                    from drama3.model.npu_recurrent import recurrent_decode

                    linear.recurrent_gated_delta_rule = recurrent_decode
        for module in model.tower.modules():
            if isinstance(module, TowerAttention):
                module.use_npu_paged_attention = use_npu_attention
        if use_npu_norm:
            from drama3.model.npu_norm import NpuGemmaRMSNorm

            for name, module in list(model.named_modules()):
                if isinstance(module, (GemmaRMSNorm, Qwen3_5MoeRMSNorm)) and module.weight.device.type == 'npu':
                    parent, leaf = name.rsplit('.', 1)
                    setattr(model.get_submodule(parent), leaf, NpuGemmaRMSNorm(module))
        if use_npu_gdn_graph:
            from drama3.model.npu_graph import GraphGDN

            for layer, device in zip(model.backbone.model.layers, layer_devices):
                if hasattr(layer, 'linear_attn') and device.type == 'npu':
                    layer.linear_attn = GraphGDN(layer.linear_attn)
        if use_npu_layer_graph:
            from drama3.model.npu_graph import GraphGDNDecoder

            for index, (layer, device) in enumerate(zip(list(model.backbone.model.layers), layer_devices)):
                if hasattr(layer, 'linear_attn') and device.type == 'npu':
                    model.backbone.model.layers[index] = GraphGDNDecoder(layer)
        first_residual = model.tower.codebook_size
        model.register_buffer(
            "backbone_code_embeddings",
            model.tower.audio_decoder.codebook_embeddings.weight[
                first_residual : 2 * first_residual
            ].to(model.devices[0]),
            persistent=False,
        )
        candidate_ids = torch.cat(
            (
                torch.tensor([model.config["eos_token_id"]], device=model.devices[-1]),
                torch.arange(
                    model.config["semantic_start_token_id"],
                    model.config["semantic_end_token_id"] + 1,
                    device=model.devices[-1],
                ),
            )
        )
        model.register_buffer("candidate_ids", candidate_ids, persistent=False)
        model.register_buffer(
            "candidate_weights",
            model.backbone.lm_head.weight.index_select(0, candidate_ids),
            persistent=False,
        )
        return model.eval()

    def embed_frames(self, frames: torch.Tensor):
        frames = frames.to(self.devices[0])
        tokens, codes = frames[..., 0], frames[..., 1:]
        embeddings = self.backbone.model.embed_tokens(tokens)
        if self.tower.backbone_codebooks != 1:
            raise NotImplementedError(
                "The reference DRAMA-3 checkpoint uses one backbone residual codebook"
            )
        residual = nn.functional.embedding(codes[..., 1], self.backbone_code_embeddings)
        mixed = (embeddings + residual) / math.sqrt(2)
        return torch.where(
            self.tower.is_semantic(tokens).unsqueeze(-1), mixed, embeddings
        )

    def hidden(self, frames, *, cache=None, attention_mask=None, position_ids=None):
        return self.backbone.model(
            inputs_embeds=self.embed_frames(frames),
            past_key_values=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

    def semantic_logits(self, hidden):
        return nn.functional.linear(hidden.to(self.devices[-1]), self.candidate_weights)
