import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from drama3.model.backbone import Drama3Model


def tiny_checkpoint(directory, *, overrides=None):
    config = {
        "eos_token_id": 2,
        "semantic_start_token_id": 64,
        "semantic_end_token_id": 79,
        "quantization_config": {
            "quant_method": "bsgemm-moe",
            "weight_block_size": [128, 128],
        },
        "text_config": {
            "model_type": "fish_omni_text",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "intermediate_size": 64,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "vocab_size": 96,
            "pad_token_id": 0,
            "max_position_embeddings": 64,
            "layer_types": ["linear_attention", "full_attention"],
            "mlp_layer_types": ["moe", "moe"],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
            },
            "layernorm_zero_centered_gamma": True,
            "tie_word_embeddings": False,
        },
        "audio_decoder_config": {
            "dim": 32,
            "n_head": 2,
            "n_local_heads": 1,
            "head_dim": 16,
            "intermediate_size": 64,
            "num_codebooks": 3,
            "vocab_size": 16,
            "shared_enrichment_layers": 1,
            "rotary_percent": 0.5,
            "vq_backbone_codebooks": 1,
        },
    }
    for section, values in (overrides or {}).items():
        config[section].update(values)
    torch.manual_seed(17)
    reference = Drama3Model(config, cache_slots=1, max_length=32).eval()
    tensors, expected = {}, {}
    for name, parameter in reference.named_parameters():
        expected[name] = parameter.detach().to(torch.bfloat16).clone()
        if name.startswith("tower."):
            tensors[name[6:]] = expected[name].contiguous()
        else:
            source = name[9:]
            if source.startswith("model."):
                source = "model.language_model." + source[6:]
            if source.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                prefix, kind = source.rsplit(".", 1)
                for expert in range(4):
                    projections = (
                        [("gate_proj", slice(0, 16)), ("up_proj", slice(16, 32))]
                        if kind == "gate_up_proj"
                        else [("down_proj", slice(None))]
                    )
                    for projection, rows in projections:
                        weight = (
                            parameter.detach()[expert, rows]
                            .to(torch.float8_e4m3fn)
                            .contiguous()
                        )
                        tensors[f"{prefix}.{expert}.{projection}.weight"] = weight
                        tensors[f"{prefix}.{expert}.{projection}.weight_scale_inv"] = (
                            torch.ones(1, 1)
                        )
                        expected[name][expert, rows] = weight.to(torch.bfloat16)
            else:
                tensors[source] = expected[name].contiguous()
    save_file(tensors, str(directory / "test.safetensors"))
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "test.safetensors" for name in tensors}})
    )
    return expected


class ModelTests(unittest.TestCase):
    def test_checkpoint_mapping_and_cached_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            expected = tiny_checkpoint(directory)
            model = Drama3Model.load(directory, ["cpu"], max_length=32)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, expected[name], atol=0, rtol=0)
            frames = torch.tensor(
                [[[1, 0, 0, 0], [3, 0, 0, 0], [64, 0, 1, 2], [65, 1, 2, 3]]]
            )
            with torch.inference_mode():
                full = model.hidden(frames).last_hidden_state[:, -1]
                prefix = model.hidden(frames[:, :-1])
                incremental = model.hidden(
                    frames[:, -1:], cache=prefix.past_key_values
                ).last_hidden_state[:, -1]
                torch.testing.assert_close(incremental, full, atol=0.03, rtol=0.03)
                selected = model.semantic_logits(full)
                dense = model.backbone.lm_head(full).index_select(
                    -1, model.candidate_ids
                )
                torch.testing.assert_close(selected, dense, atol=0.01, rtol=0.01)


if __name__ == "__main__":
    unittest.main()
