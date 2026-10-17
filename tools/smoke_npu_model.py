"""Synthetic NPU state/precision smoke; not a real-model quality benchmark."""

from pathlib import Path
import tempfile
import torch
import torch_npu  # noqa: F401 - registers the NPU device backend

from drama3.model.backbone import Drama3Model
from drama3.model.tower import TowerBatch
from tests.test_model import tiny_checkpoint

torch.npu.set_device(0)
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    tiny_checkpoint(root)
    model = Drama3Model.load(root, ["npu:0"], max_length=32)
    frames = torch.tensor(
        [[[1, 0, 0, 0], [3, 0, 0, 0], [64, 0, 1, 2], [65, 1, 2, 3]]], device="npu"
    )
    with torch.inference_mode():
        full = model.hidden(frames).last_hidden_state
        prefix = model.hidden(frames[:, :-1])
        incremental = model.hidden(
            frames[:, -1:], cache=prefix.past_key_values
        ).last_hidden_state[:, -1]
        torch.testing.assert_close(incremental, full[:, -1], atol=0.03, rtol=0.03)
        selected = model.semantic_logits(incremental)
        dense = model.backbone.lm_head(incremental).index_select(
            -1, model.candidate_ids
        )
        torch.testing.assert_close(selected, dense, atol=0.01, rtol=0.01)
        teacher_codes = model.tower.run(
            full[0, :-1],
            frames[0, :-1, 1:],
            model.tower.is_semantic(frames[0, :-1, 0]),
            frames[0, 1:, 0],
            frames[0, 1:, 1:],
            TowerBatch.prefill(0, 3, frames.device),
            (1.0, 8, 0.9),
        )
        codes = model.tower.run(
            full[0, -1:],
            frames[0, -1:, 1:],
            model.tower.is_semantic(frames[0, -1:, 0]),
            torch.tensor([66], device="npu"),
            None,
            TowerBatch.extend(0, 3, 4, frames.device),
            (1.0, 8, 0.9),
        )
        torch.npu.synchronize()
        assert codes.shape == (1, 3)
        assert ((codes >= 0) & (codes < 16)).all().item()
        print(
            {
                "cached_hidden_max_abs_error": (incremental - full[:, -1])
                .abs()
                .max()
                .item(),
                "selected_logits_max_abs_error": (selected - dense).abs().max().item(),
                "generated_code_shape": list(codes.shape),
            },
            flush=True,
        )
print("SYNTHETIC_NPU_SMOKE_PASS", flush=True)
