# DRAMA-3 Ascend campaign: incomplete

The real-model comparison has not run. No throughput, TTFA, RTF, speech-quality
or 2x improvement claim is supported by this checkpoint of the work.

The supplied model directory is `/dataset/models/drama-3`. All eight shards,
codec, index, tokenizer and metadata now pass the full manifest audit for
`e0a9218aa48888270f06ef1ed9ea67622c0e17d2`; both download and audit exited 0.
See `checkpoint-audit.json` and `download-missing-stable.log`.

## Frozen source checkouts

| Component | Revision | Checkout |
|---|---|---|
| vLLM-Omni main | `6d29fde51c08b2512eafa8778315c2eda745bebe` | `/home/why/drama3-research/vllm-omni-main-20261016` |
| SGLang-Omni main | `2c23b68f4f763c14cfb845eadb4883a5e791b701` | `/home/why/drama3-research/sglang-omni-main-20261016` |
| OmniRSI PR #9 knowledge | `4ad551a916678a177d1ffc035a972bebd5b34c39` | `/home/why/drama3-research/OmniRSI` |

A separate current-main NPU port lives in
`/home/why/drama3-research/sglang-omni-npu-20261017`. It is experimental and is
not the unmodified main baseline. On 910B1 the native GDN prefill rejects FP32
state. The installed `sgl_kernel_npu` alternative passed a synthetic CPU
recurrence comparison with FP32 state (`gdn-prefill-npu.log`).

The native decode attention produced NaN at `tower_attention.3`. Earlier
legal-code-ID probes must NOT be treated as correctness passes: NaN sampling
had yielded all-zero residual codes. FIA also rejects this model's head_dim 256.
Replacing decode attention with exact pooled-KV math SDPA produced finite
synthetic frames and exit 0 (`sglang-current-main-dense-attention-probe.log`).
This is a diagnostic, not an optimized baseline or real-model quality result.
The diagnostic scripts preserve all three paths for reproduction.

Current vLLM main imports the model class but the process aborts with exit 134
(`vllm-current-main-import.log`). Its model currently rejects TP/PP > 1. Neither
baseline is ready for matched measurements on this machine.

The CPU + physical NPU 7 smoke generated 32 finite acoustic frames and finite
nonzero PCM audio, then exited 0. It stopped at the frame limit (1.486 s audio),
so it does not establish complete-utterance quality. Its timings are ineligible
for the NPU comparison. Physical NPU 4 subsequently became idle; the two-NPU
grouped-MoE smoke on 4 and 7 produced 76 frames (3.529 s), terminated at EOS
and exited 0. Whisper transcription is diagnostic only; full quality is pending.

## Completed work

- Unified raw PCM streaming benchmark client, preserving per-request audio,
  timestamps, HTTP/transport failures and warmup traces.
- Server acoustic-frame telemetry joins by request ID. Missing telemetry leaves
  frame throughput undefined. PCM samples and generic token counts are not
  substituted for acoustic frames.
- Resumable downloader supports a configurable mirror and Python 3.9 HTTP 308
  redirects. An integration test verifies that Range survives the redirect.
- Standalone checkpoint auditor verifies all manifest files, including metadata.
- Reference streaming HTTP service with bounded queues, cancellation and frame telemetry.
- 23 tests passed in the existing container, with exit 0 (`unit-tests-server-id.log`).
  These are unit/client/reference-model tests, not full-model NPU validation.

## Measurement contract to finish before optimization

Use the same two 910B1 cards (only when free), model hashes, precision, prompt
corpus, seed/sampling, maximum frames and streaming windows for all engines.
The stronger baseline is selected separately for concurrency 1/2/4/8/16/32/64.
Keep native deployment defaults as a separate result if streaming windows differ.
No final workload or quality thresholds have been frozen yet.

TTFA is client send to first complete PCM sample. RTF is send to last audio
arrival divided by emitted audio duration. Throughput is verified successful
acoustic frames divided by the entire measured workload interval, including
failed requests. Preserve request-level samples for p50/p95; do not average
round p95s. Warmup is excluded. Repeat trials in alternating order and record
server shutdown separately from client completion.

## Resume commands

Do not start a second downloader while the current one is running.

```sh
python tools/download_checkpoint.py \
  --tree /home/why/drama3-research/model-tree.json \
  --destination /dataset/models/drama-3 \
  --repo fishaudio/drama-3 \
  --revision e0a9218aa48888270f06ef1ed9ea67622c0e17d2 \
  --endpoint https://hf-mirror.com --workers 16

python tools/audit_checkpoint.py \
  --tree /home/why/drama3-research/model-tree.json \
  --directory /dataset/models/drama-3 \
  --revision e0a9218aa48888270f06ef1ed9ea67622c0e17d2 \
  --output evidence/campaign-20261016/checkpoint-audit.json
```

The auditor must exit 0 before real-model execution. Use the same manifest and
revision; mirror transport does not change the integrity requirement.

Client usage after the service and workload are validated:

```sh
python -m drama3.benchmark --help
```

Workload JSONL rows contain the complete speech request, including
`stream: true` and `response_format: "pcm"`. Use the same file for each engine.
Frame sidecar JSONL rows contain `request_id` and `generated_frames` emitted by
verified server instrumentation, excluding EOS and codec overlap. The client
sets `X-Request-Id`; the client uses the response X-Request-Id if the server assigns its own ID,
otherwise the sidecar must preserve the supplied client ID. A client
run alone always remains `INCONCLUSIVE` pending model quality, source/contract
matching and clean server exit evidence.

## Upstream workload compatibility

The pinned SGLang cookbook uses Seed-TTS English, pre-encoded single voice
references, temperature 1.0 / top_p 0.9 / top_k 30, max_new_tokens 500.
`tools/prepare_seed_workload.py` prepares the first 128 local English rows using
shared FP32 CPU reference encoding; the manifest records source audio, metadata,
codec and workload hashes. This is a performance subset, not full accuracy.

Use `--adapt-drama3-request` with the common workload: SGLang receives
`references[].vq_codes`, vLLM and this engine receive `ref_audio_codes`. The client
retains wire payloads and explicitly selects raw audio instead of vLLM's default
SSE. vLLM flat speech sampling fields are rejected; verify matching deployed
sampling defaults. Current vLLM deploy uses BF16 GDN state and different codec
chunk sizes, so defaults do not satisfy the matched precision/streaming contract.
Those configurations and quality thresholds must be frozen before timing.

## Real-weight and native-attention follow-up

The two-NPU independent engine emitted 76 frames and EOS, with clean exit 0.
Local Whisper-large-v3 transcribed the whole clip as the supplied sentence.
The HTTP C2 smoke (one warmup, two measured requests) also exited cleanly at both
client and server; 43 and 61 model frames match the retained PCM exactly. The
first measured clip transcribes to its target text. These are functionality
checks, not a full quality evaluation or a statistically qualified C2 benchmark.
The `http-c2-smoke/results.json` overall verdict stays INCONCLUSIVE; its nested
metric eligibility flag only describes complete timing/frame data.

The SGLang NaN investigation isolated one-token KV pages: the installed native
ATB paged attention leaves NaN-filled output unwritten for page_size=1, whereas
16/128-token pages agree with explicit softmax attention within BF16 rounding
(max absolute difference <= 0.00815 in the retained fixture). The corrected
layout test is `paged-attention-layouts-corrected.log`; the earlier first test
had a mismatched reference length for page_size=1 and is not parity evidence.

The experimental current-main port now uses page_size=16 and binds the FP32
GDN alternate kernel per backend instance. Its synthetic scheduler, using native
attention and no external attention monkeypatch, exits 0 with the same frames
as the math diagnostic (`sglang-integrated-page16-probe.log`). Its actual
TP2 HTTP deployment is still under validation; `sglang-page16-npu-wip.patch`
records this unqualified port, not a published or completed baseline.

### SGLang real-weight TP2 HTTP smoke

Experimental main-based NPU port served two C2 requests with 42 and 73 acoustic frames, both EOS; PCM lengths equal frames × 2048 samples. The first independent Whisper transcript matches the prompt. Artifacts: `sglang-c2-smoke/`. This is not a formal baseline: one warmup request does not cover C2, quality qualification is incomplete, and shutdown failed in the TP follower (Gloo peer disconnect followed by heap corruption), despite root exit 0. Overall INCONCLUSIVE.

### TP shutdown regression

The experimental scheduler now broadcasts a shutdown marker and exits all TP ranks at the same collective boundary. A real two-process Gloo regression (`tools/probe_sglang_tp_shutdown.py`) passed with exit 0. All repository pre-commit checks passed. The repeated real-model C2 smoke on physical cards 0 and 1 used two warmup requests, produced 44 and 65 measured frames with exact PCM lengths, and root exited 0. Both TP schedulers stopped; no Gloo disconnect, heap corruption or forced termination appeared. A resource_tracker shared-memory warning remains, and individual worker exit codes are not instrumented: lifecycle remains INCONCLUSIVE. See `sglang-c2-shutdown-regression/` and `sglang-attempt5-manifest.json`. This changed hardware cohort is not a performance A/B. An initial client call before readiness failed during warmup and did not start measurement; its log and traces are retained separately.
