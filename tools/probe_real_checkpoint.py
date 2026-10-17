"""Real-weight correctness probe; CPU/NPU placement is not a serving baseline.

Pass the same arguments as tools/generate.py. Requires a completed pinned audit.
Sampling and codec output are checked before invalid values can be hidden by
argmax, clipping, or integer PCM conversion.
"""

import argparse
import json
from pathlib import Path
import runpy

import torch

from drama3.model import tower
from drama3.model.codec import AudioCodec


def main():
    root = Path(__file__).resolve().parents[1]
    audit = json.loads(
        (root / "evidence/campaign-20261016/checkpoint-audit.json").read_text()
    )
    if not audit["complete"] or audit["status"] != "PASS":
        raise RuntimeError("A complete checkpoint integrity audit is required")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", type=Path, required=True)
    args, _ = parser.parse_known_args()
    if args.model.resolve() != Path(audit["model_directory"]).resolve():
        raise RuntimeError("Requested model differs from the audited directory")
    original_sample = tower.sample_rows
    original_decode = AudioCodec.decode

    def checked_sample(logits, *args, **kwargs):
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("Nonfinite real-model sampling logits")
        return original_sample(logits, *args, **kwargs)

    def checked_decode(self, codes):
        audio = original_decode(self, codes)
        if not torch.isfinite(audio).all().item():
            raise RuntimeError("Nonfinite real-codec audio")
        if audio.abs().max().item() == 0:
            raise RuntimeError("Real codec returned all-zero audio")
        return audio

    tower.sample_rows = checked_sample
    AudioCodec.decode = checked_decode
    runpy.run_path(str(root / "tools/generate.py"), run_name="__main__")
    print("REAL_CHECKPOINT_SMOKE_PASS_NOT_QUALITY_OR_PERFORMANCE", flush=True)


if __name__ == "__main__":
    main()
