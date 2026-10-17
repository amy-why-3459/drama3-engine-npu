"""Strict checkpoint loading and acoustic-frame decoding for DRAMA-3."""

from pathlib import Path

import torch
from torch import nn

from drama3.model.dac_utils import DAC_HOP_LENGTH, DAC_SAMPLE_RATE, build_dac_codec


class AudioCodec(nn.Module):
    sample_rate = DAC_SAMPLE_RATE
    hop_length = DAC_HOP_LENGTH

    def __init__(self, checkpoint: Path, device: str, dtype: torch.dtype):
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        with torch.device("meta"):
            self.codec = build_dac_codec()
        self.codec.load_state_dict(state, strict=True, assign=True)
        self.codec.eval().requires_grad_(False)
        self.codec.to(device=device, dtype=dtype)

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode [batch, 10 codebooks, frames] into [batch, samples].

        The caller supplies valid codec indices (semantic <4096, residual <1024).
        Frame count is explicit; no token-to-audio estimate is used.
        """
        if codes.ndim != 3 or codes.shape[1] != 10 or codes.shape[2] == 0:
            raise ValueError("Expected nonempty [batch, 10, frames] codec indices")
        lengths = torch.full(
            (codes.shape[0],), codes.shape[2], device=codes.device, dtype=torch.long
        )
        audio, _ = self.codec.decode(codes, lengths)
        if audio.shape[-1] != codes.shape[-1] * self.hop_length:
            raise RuntimeError("Codec output does not match the acoustic frame count")
        return audio[:, 0].float()
