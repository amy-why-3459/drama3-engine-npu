"""Streaming audio metrics from client-observed timestamps and verified frame counts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class RequestTrace:
    request_id: str
    sent_s: float
    first_audio_s: float | None
    last_audio_s: float | None
    terminal_s: float
    audio_samples: int
    sample_rate: int
    generated_frames: int | None
    success: bool
    error: str | None = None
    server_request_id: str | None = None

    def validate(self) -> None:
        times = [self.sent_s, self.terminal_s]
        times.extend(
            t for t in (self.first_audio_s, self.last_audio_s) if t is not None
        )
        if not all(math.isfinite(t) for t in times) or self.terminal_s < self.sent_s:
            raise ValueError(f"{self.request_id}: invalid timestamps")
        if self.audio_samples < 0 or self.sample_rate <= 0:
            raise ValueError(f"{self.request_id}: invalid audio geometry")
        if self.generated_frames is not None and (
            isinstance(self.generated_frames, bool)
            or not isinstance(self.generated_frames, int)
            or self.generated_frames < 0
        ):
            raise ValueError(f"{self.request_id}: invalid acoustic frame count")
        if self.success:
            if (
                self.error
                or self.audio_samples == 0
                or self.first_audio_s is None
                or self.last_audio_s is None
            ):
                raise ValueError(
                    f"{self.request_id}: successful request has no complete audio"
                )
            if (
                not self.sent_s
                <= self.first_audio_s
                <= self.last_audio_s
                <= self.terminal_s
            ):
                raise ValueError(f"{self.request_id}: unordered audio timestamps")


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    left, right = math.floor(position), math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def summarize(traces: list[RequestTrace], expected_requests: int) -> dict:
    if expected_requests <= 0 or len(traces) != expected_requests:
        raise ValueError(
            "Every scheduled request, including failures, must have a trace"
        )
    if len({t.request_id for t in traces}) != len(traces):
        raise ValueError("Duplicate request ids")
    for trace in traces:
        trace.validate()
    elapsed_s = max(t.terminal_s for t in traces) - min(t.sent_s for t in traces)
    if elapsed_s <= 0:
        raise ValueError("Nonpositive measurement interval")
    good = [t for t in traces if t.success]
    ttfa_ms = [(t.first_audio_s - t.sent_s) * 1000 for t in good]
    rtfs = [
        (t.last_audio_s - t.sent_s) / (t.audio_samples / t.sample_rate) for t in good
    ]
    frames_known = all(
        t.generated_frames is not None and t.generated_frames > 0 for t in good
    )
    return {
        "requests": len(traces),
        "successful_requests": len(good),
        "failed_requests": len(traces) - len(good),
        "wall_time_s": elapsed_s,
        "throughput_frames_s": sum(t.generated_frames for t in good) / elapsed_s
        if good and frames_known
        else None,
        "audio_seconds_per_wall_second": sum(
            t.audio_samples / t.sample_rate for t in good
        )
        / elapsed_s,
        "ttfa_p50_ms": percentile(ttfa_ms, 50),
        "ttfa_p95_ms": percentile(ttfa_ms, 95),
        "rtf_mean": fmean(rtfs) if rtfs else None,
        "eligible_for_comparison": len(good) == expected_requests and frames_known,
    }


def compare(
    candidate: dict, baselines: list[dict], required_speedup: float = 2.0
) -> dict:
    """Reject unmatched conditions or missing quality before computing a speedup."""
    if not math.isfinite(required_speedup) or required_speedup <= 0:
        raise ValueError("required_speedup must be finite and positive")
    if len(baselines) != 2 or {b.get("engine") for b in baselines} != {
        "vllm-omni",
        "sglang-omni",
    }:
        return {
            "verdict": "INCONCLUSIVE",
            "reason": "Both distinct framework baselines are required",
        }
    runs = [candidate, *baselines]
    required_contract = {
        "model_revision",
        "model_hashes",
        "hardware",
        "device_count",
        "workload_sha256",
        "concurrency",
        "sampling",
        "frame_count_convention",
        "quality_contract_sha256",
        "precision",
        "streaming_policy",
    }
    contract = candidate.get("contract", {})
    if not required_contract <= contract.keys() or any(
        contract[k] is None for k in required_contract
    ):
        return {"verdict": "INCONCLUSIVE", "reason": "Incomplete experiment contract"}
    if any(r.get("contract") != contract for r in baselines):
        return {"verdict": "INCONCLUSIVE", "reason": "Experiment contracts differ"}
    for run in runs:
        if not run.get("source_revision") or not run.get("clean_exit"):
            return {
                "verdict": "INCONCLUSIVE",
                "reason": "Missing source revision or clean process exit",
            }
        if run.get("quality") != "PASS" or not run.get("quality_evidence"):
            return {
                "verdict": "INCONCLUSIVE",
                "reason": "Missing passing real-model quality evidence",
            }
        metrics = run.get("metrics", {})
        rate = metrics.get("throughput_frames_s")
        if (
            not metrics.get("eligible_for_comparison")
            or not isinstance(rate, (float, int))
            or not math.isfinite(rate)
            or rate <= 0
        ):
            return {
                "verdict": "INCONCLUSIVE",
                "reason": "Incomplete or invalid performance measurements",
            }
    best = max(baselines, key=lambda b: b["metrics"]["throughput_frames_s"])
    speedup = (
        candidate["metrics"]["throughput_frames_s"]
        / best["metrics"]["throughput_frames_s"]
    )
    return {
        "verdict": "PASS" if speedup >= required_speedup else "FAIL",
        "speedup": speedup,
        "stronger_baseline": best["engine"],
        "required_speedup": required_speedup,
    }
