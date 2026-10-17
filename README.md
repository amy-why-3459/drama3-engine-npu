# DRAMA-3 on Ascend

Standalone PyTorch DRAMA-3 execution and continuous batched PCM streaming on Ascend 910B1. Full-weight C4 validation completed 4 warmup and 4 measured requests, with client/server exit 0 and matching acoustic-frame accounting. The four measured outputs have WER 0 and mean SIM 59.94. This is functional smoke coverage, not a qualified performance or full-quality benchmark; the requested 2x speedup remains unverified.

Latest validated model revision: `5a74c2e`. Evidence and current optimization status: [`evidence/campaign-20261017/README.md`](evidence/campaign-20261017/README.md).

The accepted target is 2x acoustic-frame throughput over the faster of pinned vLLM-Omni and SGLang-Omni NPU-adapted baselines at each concurrency 1, 2, 4, 8, 16, 32, 64. All engines must use matched hardware, checkpoint, requests, sampling, streaming policy, and real-model quality evaluation. Raw main and NPU port revisions are tracked separately.

Current components:

- `drama3/serving.py`: PCM streaming reference service for text and pre-encoded voice references, with interleaved eager requests, request-owned cache slots, bounded output queues, cancellation cleanup and acoustic-frame telemetry. The opt-in `--batched` path groups active requests into one backbone decode and one tower decode, with independent request state. Full-weight C4 smoke coverage passes; matched baseline performance and full-quality qualification remain pending.
- `drama3/benchmark.py`: shared HTTP load generator retaining PCM, warmup and measured traces, failures, and explicit frame-sidecar joins. Server lifecycle and real speech quality remain separate acceptance gates.
- `drama3/model`: independent checkpoint loader, Transformers Qwen3.5-MoE backbone, layer placement, and DRAMA-3 prompt/codebook tower. No vLLM/SGLang runtime imports. All checkpoint files pass integrity validation; real two-NPU generation passes a smoke test. Full quality validation remains pending.
- `drama3/batching.py`: continuous batched decode with variable-length hybrid KV/GDN state, arrivals, EOS, request reordering and slot reuse. `--npu-attention` reads the tower cache through native 16-token pages. CPU regressions and synthetic native-attention parity pass; these do not establish real-model quality or speedup.
- `drama3/generation.py`: eager reference generation with request-owned tower state, semantic RAS, EOS termination and early-close cleanup. `tools/generate.py` writes codes, WAV and timing diagnostics; it is not a serving benchmark.
- `drama3/model/codec.py`: strict DAC checkpoint loading and explicit acoustic-frame decoding. Real codec weights passed a synthetic-index NPU smoke test; generated-speech quality is still unmeasured.
- `drama3/metrics.py`: client-timestamp metrics and fail-closed comparison. Acoustic frames must be reported by verified instrumentation, not inferred from generic token counts or PCM duration. RTF ends at the last audio sample; throughput uses the complete measured workload interval, including failures.
- `tools`: checkpoint downloader with resumable byte ranges and SHA256 verification; backend and layout preflights.
- `evidence/preflight`: preliminary logs. Constructor tests and synthetic checks are not full-model performance evidence.

The initial checkpoint is block-FP8 MoE with BF16 non-expert weights. Exact block expansion into BF16 is the first reference path for Ascend 910B; it needs more than one card for full residency. Device allocation, quality gates and workload will be frozen before performance search.

## Tests

From this directory, in the existing Python 3.12 / Torch environment:

```sh
python -m unittest discover -s tests -v
```

Client/service tests also require the `benchmark` optional dependency (`aiohttp`).

## Streaming service

After the full checkpoint passes `tools/audit_checkpoint.py`, run in the existing
NPU environment, selecting only available physical devices:

```sh
ASCEND_RT_VISIBLE_DEVICES=4,7 python -m drama3.serving \
  --model /dataset/models/drama-3 --devices npu:0 npu:1 --npu-moe \
  --max-active 8 --context-length 2048 \
  --telemetry runs/reference-frames.jsonl
```

Add `--batched --npu-attention` to select the batched implementation with native tower decode attention. The reference path remains available for matched correctness checks. Experimental `--npu-layer-graph` captures complete GDN decoder blocks including grouped MoE; it requires `--npu-moe`. `--npu-gdn-graph` captures only the GDN module and cannot be combined with the layer option. Both are tested on dual-NPU tiny models with state/padding parity, but full-weight graph qualification is pending. Optional `--npu-norm` selects fused zero-centered RMSNorm, with component and two-NPU tiny-model parity coverage. `--npu-gdn-decode` selects the standalone Triton FP32-state single-token recurrence. Optional `--npu-gdn-prefill` uses the installed `sgl_kernel_npu.fla` kernel library with FP32 recurrent state; it does not import the SGLang serving runtime. This flag requires that kernel package in the Python environment. Tiny-model NPU batching and synthetic FP32-state prefill checks pass; full-weight C4 smoke coverage is recorded in the campaign evidence.

`POST /v1/audio/speech` accepts `input`, `stream: true`, `response_format: "pcm"`,
optional `max_new_tokens` (acoustic frames), `temperature`, `top_k`, and `top_p`.
One pre-encoded voice reference is accepted as `ref_audio_codes` ([10, frames]
integer codec indices) and `ref_text`; reference audio upload/encoding is not
part of this endpoint.
Output is mono PCM16 at 44.1 kHz. Use a fresh telemetry path per server process;
`X-Request-Id` joins client measurements to server frame counts. Chunk scheduling
defaults to 2 initial frames, then 25 frames with 25 left-context frames. These
parameters must match the baselines before comparison. The service uses the
reference model and process-level RNG seed; reproducible quality evaluation
must account for sampling under interleaved requests.

The reference server has unit, mocked HTTP, and real-weight C2 smoke coverage.
The opt-in `--npu-moe` uses grouped BF16 expert matmuls. Real smoke audio has
independent ASR transcripts, but full Seed-TTS quality and baseline comparison
remain pending. See
[`evidence/campaign-20261016/README.md`](evidence/campaign-20261016/README.md).

The benchmark's `--adapt-drama3-request` maps this single-reference payload to
SGLang's `references[].vq_codes` and explicitly selects vLLM's raw audio stream
(its current default is SSE). It retains the exact wire payloads separately
from the common workload hash. The pinned vLLM speech schema does not expose
flat `temperature`/`top_k`/`top_p`; the adapter rejects those fields for vLLM.
Freeze and verify identical server sampling defaults when omitting these fields
from the shared workload. Native endpoint payloads remain supported without
the adapter flag.

The base metric module uses only the Python standard library; model tests need PyTorch, Transformers and safetensors. Do not install CUDA dependencies into the shared NPU environment. Baseline SGLang dependencies are being isolated in a separate virtual environment.

## Provenance

The prompt, configuration and codebook reference derive from Apache-2.0 vLLM-Omni source at `41ed44cc014dfe0a46499ffd63473addf05777f5`; original notices remain in each derived file and the license is in `licenses/`. The standalone tower currently uses portable reference attention instead of CUDA graph/FlashAttention integration.

DAC modules derive from the same pinned vLLM-Omni tree, retaining its Fish Speech and Descript notices. The Descript MIT license is included separately. The codec checkpoint SHA256 is `74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f`.

Research guidance: OmniRSI PR #9 `4ad551a916678a177d1ffc035a972bebd5b34c39`, specifically Strata design/validation and the two Omni serving knowledge packs. KDA is used as an experiment workflow; its CUDA/B200/B300 kernels are not claimed to run on Ascend.
