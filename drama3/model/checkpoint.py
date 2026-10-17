"""Strict CPU FP8 block expansion and checkpoint access for DRAMA-3."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


def expand_block_fp8(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size=(128, 128),
    dtype=torch.bfloat16,
):
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("Expert weight and scales must both be matrices")
    rows, columns = weight.shape
    block_rows, block_columns = block_size
    if block_rows <= 0 or block_columns <= 0:
        raise ValueError("Block dimensions must be positive")
    if scale.shape != (
        (rows + block_rows - 1) // block_rows,
        (columns + block_columns - 1) // block_columns,
    ):
        raise ValueError("Scale shape does not match the FP8 weight blocks")
    if weight.device.type != "cpu" or scale.device.type != "cpu":
        raise ValueError(
            "Checkpoint expansion is performed on CPU before device placement"
        )
    output = torch.empty((rows, columns), dtype=dtype)
    for start in range(0, rows, block_rows):
        end = min(start + block_rows, rows)
        gains = (
            scale[start // block_rows]
            .float()
            .repeat_interleave(block_columns)[:columns]
        )
        output[start:end] = (weight[start:end].float() * gains).to(dtype)
    return output


class Checkpoint:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.config = json.loads((self.directory / "config.json").read_text())
        self.index = json.loads(
            (self.directory / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.loaded_names: set[str] = set()
        quantization = self.config["quantization_config"]
        if quantization["quant_method"] != "bsgemm-moe" or quantization[
            "weight_block_size"
        ] != [128, 128]:
            raise ValueError(
                "Only the original DRAMA-3 block-FP8 checkpoint is supported"
            )
        self.readers = {}
        try:
            for shard in sorted(set(self.index.values())):
                if Path(shard).is_absolute() or ".." in Path(shard).parts:
                    raise ValueError("Invalid checkpoint shard path")
                self.readers[shard] = safe_open(
                    str(self.directory / shard), framework="pt", device="cpu"
                )
        except Exception:
            self.close()
            raise

    def tensor(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        weight = self.readers[self.index[name]].get_tensor(name)
        self.loaded_names.add(name)
        scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
        if weight.dtype == torch.float8_e4m3fn:
            if not name.endswith(".weight") or scale_name not in self.index:
                raise ValueError(f"Missing block scales for {name}")
            scale = self.readers[self.index[scale_name]].get_tensor(scale_name)
            self.loaded_names.add(scale_name)
            return expand_block_fp8(weight, scale, dtype=dtype)
        if scale_name in self.index:
            raise ValueError(f"Unexpected scale for non-FP8 parameter {name}")
        return weight.to(dtype)

    def check_complete(self):
        missing = set(self.index) - self.loaded_names
        if missing:
            raise ValueError(
                f"{len(missing)} unused checkpoint tensors: {sorted(missing)[:8]}"
            )

    def close(self):
        for reader in self.readers.values():
            reader.__exit__(None, None, None)
        self.readers.clear()
