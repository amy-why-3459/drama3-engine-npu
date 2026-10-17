"""Shared PCM streaming load generator; model frame counts require a sidecar.

Run with ``python -m drama3.benchmark --help``. No inference runtime is imported.
The retained PCM and request traces are inputs to independent quality evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from drama3.metrics import RequestTrace, summarize


def adapt_drama3_request(payload, engine):
    """Map a common single-reference request without changing codec indices.

    Sampling must be frozen in the server configurations when omitted. vLLM's
    speech API does not expose these three flat sampling fields at the pinned
    revision; reject them rather than silently relying on ignored parameters.
    """
    result = dict(payload)
    if engine == 'vllm-omni':
        if {'temperature', 'top_k', 'top_p'} & result.keys():
            raise ValueError('Pin vLLM sampling in deployment; flat speech sampling fields are unsupported')
        result['stream_format'] = 'audio'
    elif engine == 'sglang-omni' and 'ref_audio_codes' in result:
        if 'references' in result or not result.get('ref_text'):
            raise ValueError('Ambiguous reference or missing reference transcript')
        result['references'] = [{'vq_codes': result.pop('ref_audio_codes'),
                                 'text': result.pop('ref_text')}]
    return result


def attach_frames(traces, records):
    """Join server instrumentation by request ID, never by completion order."""
    indexed = {}
    for record in records:
        key = record["request_id"]
        if key in indexed:
            raise ValueError(f"Duplicate frame telemetry: {key}")
        count = record["generated_frames"]
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid frame telemetry: {key}")
        indexed[key] = count
    ids = [t.server_request_id or t.request_id for t in traces]
    if len(ids) != len(set(ids)):
        raise ValueError('Server returned ambiguous request IDs')
    return [RequestTrace(**{**asdict(t), "generated_frames": indexed.get(t.server_request_id or t.request_id)})
            for t in traces]


async def request(session, endpoint, payload, request_id, output, sample_rate):
    first = last = None
    size = 0
    error = None
    server_request_id = None
    sent = time.perf_counter()
    try:
        with output.open("xb") as audio:
            async with session.post(endpoint, json=payload,
                                    headers={"X-Request-Id": request_id}) as response:
                server_request_id = response.headers.get('X-Request-Id')
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:512]}")
                kind = response.headers.get("Content-Type", "").split(";", 1)[0]
                if kind not in {"audio/pcm", "audio/raw", "application/octet-stream"}:
                    raise ValueError(f"Expected raw PCM, received {kind!r}")
                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    now = time.perf_counter()
                    size += len(chunk)
                    # A split byte is not yet a complete PCM16 sample.
                    if first is None and size >= 2:
                        first = now
                    last = now
                    audio.write(chunk)
                if size == 0 or size % 2:
                    raise ValueError(f"Invalid mono PCM16 byte count: {size}")
    except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        # aiohttp transport failures must remain in the measured population.
        error = f"{type(exc).__name__}: {exc}"
    return RequestTrace(request_id, sent, first, last, time.perf_counter(),
                        size // 2, sample_rate, None, error is None, error, server_request_id)


async def run_batch(session, args, payloads, directory, label, count):
    queue = asyncio.Queue()
    for index in range(count):
        queue.put_nowait(index)
    traces = []

    async def worker():
        while not queue.empty():
            index = queue.get_nowait()
            request_id = f"{args.run_id}-{label}-{index:06d}"
            trace = await request(session, args.endpoint, payloads[index % len(payloads)],
                                  request_id, directory / f"{request_id}.pcm", args.sample_rate)
            traces.append(trace)

    await asyncio.gather(*(worker() for _ in range(min(args.concurrency, count))))
    return sorted(traces, key=lambda t: t.request_id)


async def run(args):
    import aiohttp

    raw = args.workload.read_bytes()
    payloads = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not payloads or any(p.get("response_format") != "pcm" or p.get("stream") is not True
                           for p in payloads):
        raise ValueError("Every workload row must request stream=true, response_format=pcm")
    if args.adapt_drama3_request:
        payloads = [adapt_drama3_request(p, args.engine) for p in payloads]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'wire-requests.json').write_text(json.dumps(payloads, indent=2) + '\n')
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout,
                                     connector=aiohttp.TCPConnector(limit=args.concurrency)) as session:
        warmup = await run_batch(session, args, payloads, args.output, "warmup", args.warmup)
        (args.output / "warmup.json").write_text(json.dumps([asdict(t) for t in warmup], indent=2))
        if any(not t.success for t in warmup):
            raise RuntimeError("Warmup failed; retained traces, measurement not started")
        traces = await run_batch(session, args, payloads, args.output, "measured", args.requests)
    (args.output / "traces.json").write_text(
        json.dumps([asdict(t) for t in traces], indent=2) + "\n"
    )
    if args.frame_telemetry:
        records = [json.loads(line) for line in args.frame_telemetry.read_text().splitlines() if line.strip()]
        traces = attach_frames(traces, records)
    result = {
        "engine": args.engine, "source_revision": args.source_revision,
        "concurrency": args.concurrency, "workload_sha256": hashlib.sha256(raw).hexdigest(),
        "run_id": args.run_id, "endpoint": args.endpoint,
        "request_adapter": "drama3-v1" if args.adapt_drama3_request else None,
        "metrics": summarize(traces, args.requests),
        "quality": "NOT_RUN", "clean_exit": None,
        "status": "INCONCLUSIVE", "traces": [asdict(t) for t in traces],
        "notes": "Client completion does not establish server clean exit or audio quality. "
                 "Frame telemetry must count emitted model acoustic frames excluding EOS and context overlap.",
    }
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))
    return 1 if result["metrics"]["failed_requests"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--engine", choices=["vllm-omni", "sglang-omni", "drama3-engine"], required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--adapt-drama3-request", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-telemetry", type=Path)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if min(args.concurrency, args.requests, args.sample_rate) <= 0 or args.warmup < 1 or args.timeout <= 0:
        parser.error("Counts, rate, timeout and warmup must be positive")
    if args.requests < args.concurrency:
        parser.error("Requests must cover the requested concurrency")
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        parser.error("run-id must contain only letters, digits, hyphens or underscores")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
