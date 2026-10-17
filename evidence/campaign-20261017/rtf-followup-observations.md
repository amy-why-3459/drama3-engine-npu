# RTF optimization follow-up observations

Targets remain unmet: mean request RTF < 0.1 at C1 and < 1 at C64.
These are shared-host, single-round diagnostics, not a matched before/after qualification.

| Local code | C1 RTF | C64 RTF | Measured requests | Server exit | Quality |
|---|---:|---:|---|---:|---|
| e3bac8a, no graphs, prefill batch 4 | 7.1543 | 27.1301 | 8 / 64 | 0 | C1 WER 0, SIM 64.3729; C64 WER 0.0071124, SIM 62.5371 |
| 80a3b86, graphs + conv/cache/DAC changes | 3.6357 | 18.5322 | 8 / 64 | -11 | Not run; lifecycle failure |
| 2d22d59, C1 cleanup isolation | 2.8280 | Not measured | 1 / none | 0 | Not run |
| 2d22d59, expanded tower graphs | 2.5804 | Invalid: 63/64 succeeded | 8 / 64 | -11 | Not run |

The single-request C1 run isolates cleanup and must not be compared as an eight-request performance result. The 8/64 campaign for 2d22d59 is recorded separately under `rtf-expanded-tower-candidate/` and failed: one request was rejected by the generated-code guard, and the process also crashed during shutdown. The guard was corrected in `2b6085f` to match the upstream DAC normalization of residual model indices; the lifecycle issue remains independent.

The full-weight 80a3b86 process crashed in `torch_npu._npu_shutdown` after the backend cleanup marker. Device-local entry destruction did not resolve it. A 400-graph worker-thread lifetime test exited 0, as did the full-weight C1-only 2d22d59 isolation run. Neither establishes that C64 cleanup is safe.

Expanded row graphs include QKV projections, Q/K normalization, RoPE, and output gating/projection. KV writes and dynamic-length attention remain outside capture. Real-shape component ABBA measurements are in `tower-expanded-graph-abba.log`; cache and mixed-arrival generation parity tests exited 0. The initial generation probe had stale helper arguments; its failure log is retained alongside the corrected probe.

W8A8 experts remain an exploratory tool only. Synthetic real-shape B1/B8 were slower and relative L2 differences were 2.6–2.8%; B64 was faster. No model-quality acceptance or runtime integration is claimed.

No optimization from this follow-up has been pushed to the public PR. Raw audio/model artifacts remain local.
