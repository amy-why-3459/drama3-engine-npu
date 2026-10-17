"""Sampling configuration without importing device runtimes in the HTTP process."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Sampling:
    temperature: float = 1.0
    top_k: int = 30
    top_p: float = 0.9
    max_frames: int = 500

    def __post_init__(self):
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.max_frames < 1:
            raise ValueError('Invalid sampling parameters')
