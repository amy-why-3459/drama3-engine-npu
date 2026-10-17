"""Run real-checkpoint eager generation; this is not a concurrency benchmark."""

import argparse
import json
from pathlib import Path
import time
import wave

import torch
import torch_npu  # noqa: F401
from transformers import AutoTokenizer

from drama3.generation import ReferenceGenerator, Sampling
from drama3.model.backbone import Drama3Model
from drama3.model.codec import AudioCodec
from drama3.model.prompt import build_prompt_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", default=["npu:0", "npu:1"])
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--npu-moe", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    if args.devices[0].startswith("npu"):
        torch.npu.set_device(args.devices[0])
        torch.npu.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = json.loads((args.model / "config.json").read_text())
    prompt = build_prompt_frames(
        tokenizer,
        args.text,
        [],
        semantic_begin_id=config["semantic_start_token_id"],
        num_codebooks=config["audio_decoder_config"]["num_codebooks"],
    )
    model = Drama3Model.load(
        args.model, args.devices, max_length=len(prompt) + args.max_frames,
        use_npu_moe=args.npu_moe,
    )
    codec = AudioCodec(args.model / "codec.pth", args.devices[-1], torch.float32)
    generator = ReferenceGenerator(model)
    started = time.perf_counter()
    frames = []
    arrival_seconds = []
    for codes in generator.frames(prompt, Sampling(max_frames=args.max_frames)):
        if not (0 <= codes[0] < 4096 and ((codes[1:] >= 0) & (codes[1:] < 1024)).all()):
            raise RuntimeError(
                f"Model emitted an invalid codec frame: {codes.tolist()}"
            )
        frames.append(codes)
        arrival_seconds.append(time.perf_counter() - started)
        print(f"Generated acoustic frame {len(frames)}", flush=True)
    if not frames:
        raise RuntimeError("Model generated EOS without any acoustic frames")
    codes = torch.stack(frames).T.unsqueeze(0).to(args.devices[-1])
    audio = codec.decode(codes)[0].cpu()
    ended = time.perf_counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setparams((1, 2, codec.sample_rate, 0, "NONE", "not compressed"))
        output.writeframes(
            (audio.clamp(-1, 1) * 32767).short().numpy().astype("<i2").tobytes()
        )
    torch.save(torch.stack(frames), args.output.with_suffix(".codes.pt"))
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "kind": "offline_eager_smoke_not_serving_benchmark",
                "frames": len(frames),
                "samples": audio.numel(),
                "sample_rate": codec.sample_rate,
                "elapsed_seconds": ended - started,
                "frame_arrival_seconds": arrival_seconds,
                "seed": args.seed,
                "text": args.text,
                "devices": args.devices,
                "npu_moe": args.npu_moe,
                "termination": "frame_limit" if len(frames) == args.max_frames else "eos",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
