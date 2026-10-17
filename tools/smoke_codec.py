"""Real codec weights with synthetic indices; not speech-quality evidence."""

import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from drama3.model.codec import AudioCodec

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--device", default="npu:0")
args = parser.parse_args()
torch.set_num_threads(8)
if args.device.startswith("npu"):
    torch.npu.set_device(args.device)
codec = AudioCodec(args.checkpoint, args.device, torch.float32)
codes = torch.zeros((1, 10, 8), dtype=torch.long, device=args.device)
audio = codec.decode(codes)
assert torch.isfinite(audio).all().item()
print({"shape": list(audio.shape), "max_abs": audio.abs().max().item()}, flush=True)
print("CODEC_SYNTHETIC_INDICES_PASS", flush=True)
