"""Validate standalone parameter naming against the real checkpoint index without weights."""

import json
from pathlib import Path
import sys

import torch

from drama3.model.backbone import Drama3Model

root = Path(sys.argv[1])
config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
with torch.device("meta"):
    model = Drama3Model(config, cache_slots=1, max_length=1024)
expected = set()
for name, parameter in model.named_parameters():
    if name.startswith("tower."):
        expected.add(name.removeprefix("tower."))
        continue
    local = name.removeprefix("backbone.")
    source = (
        local.replace("model.", "model.language_model.", 1)
        if local.startswith("model.")
        else local
    )
    if local.endswith((".experts.gate_up_proj", ".experts.down_proj")):
        prefix, kind = source.rsplit(".", 1)
        for expert in range(parameter.shape[0]):
            for projection in (
                ["gate_proj", "up_proj"] if kind == "gate_up_proj" else ["down_proj"]
            ):
                expected.add(f"{prefix}.{expert}.{projection}.weight")
                expected.add(f"{prefix}.{expert}.{projection}.weight_scale_inv")
    else:
        expected.add(source)
print(
    json.dumps(
        {
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "expected_tensors": len(expected),
            "checkpoint_tensors": len(index),
            "missing": sorted(expected - set(index)),
            "unexpected": sorted(set(index) - expected),
        },
        indent=2,
    )
)
if expected != set(index):
    raise SystemExit(1)
