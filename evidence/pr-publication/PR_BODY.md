实现独立 DRAMA-3 Ascend 推理框架，支持完整权重加载、连续批处理、请求独立缓存、PCM 流式服务，以及 NPU attention、GDN、RMSNorm 和可选 decoder layer graph。模型与调度路径不依赖 vLLM/SGLang runtime；可选 GDN prefill 使用 sgl_kernel_npu 算子库。

## 验证与范围

- 双 910B1 完整权重 C4 功能验证：4 次 warmup、4 次 measured 请求全部成功，client/server exit 0；独立质量评估 WER 0、mean SIM 59.94（仅 4 条样本）。对应模型代码 `5a74c2e`。
- Graph 实现 `f022d5c`：双 NPU 小模型输出/状态/变长 batch 验证通过；完整权重 C4 请求 4/4 成功，但该 graph 版本的质量和服务退出验证尚未完成。
- 以下数据为单轮观察，不是三框架同条件完成的性能验收；2 倍吞吐目标尚未验证。

## C1 / C8 / C16 / C32 / C64 对比

Throughput 单位 frames/s；TTFA 单位 ms；RTF 为 mean，越低越好。`—` 表示未获得有效测量，不能解释为 0。

| 框架 | 并发 | Throughput | TTFA p50 | TTFA p95 | RTF mean | 状态 |
|---|---:|---:|---:|---:|---:|---|
| DRAMA-3 standalone | 1 | — | — | — | — | 未测：已有 C4 功能样本不能替代此档位 |
| vLLM-Omni | 1 | — | — | — | — | 未测：尚无该并发档位完整基线 |
| SGLang-Omni | 1 | 3.166 | 897.24 | 1109.08 | 6.839 | 128/128 请求完成；质量/生命周期未通过完整验收 |
| DRAMA-3 standalone | 8 | — | — | — | — | 未测：已有 C4 功能样本不能替代此档位 |
| vLLM-Omni | 8 | — | — | — | — | 未测：尚无该并发档位完整基线 |
| SGLang-Omni | 8 | 19.428 | 1259.06 | 2873.44 | 8.186 | 128/128 请求完成；质量/生命周期未通过完整验收 |
| DRAMA-3 standalone | 16 | — | — | — | — | 未测：已有 C4 功能样本不能替代此档位 |
| vLLM-Omni | 16 | — | — | — | — | 未测：尚无该并发档位完整基线 |
| SGLang-Omni | 16 | 24.811 | 1787.69 | 8733.08 | 13.192 | 128/128 请求完成；质量/生命周期未通过完整验收 |
| DRAMA-3 standalone | 32 | — | — | — | — | 未测：已有 C4 功能样本不能替代此档位 |
| vLLM-Omni | 32 | — | — | — | — | 未测：尚无该并发档位完整基线 |
| SGLang-Omni | 32 | 31.135 | 3682.73 | 7268.35 | 14.665 | 128/128 请求完成；质量/生命周期未通过完整验收 |
| DRAMA-3 standalone | 64 | — | — | — | — | 未测：已有 C4 功能样本不能替代此档位 |
| vLLM-Omni | 64 | — | — | — | — | 未测：尚无该并发档位完整基线 |
| SGLang-Omni | 64 | — | — | — | — | C64 失败：TP0 exit -6，出现 heap corruption；不报有效性能 |

## 可比性与来源

- SGLang 上游 main 固定为 `2c23b68f4f763c14cfb845eadb4883a5e791b701`，NPU 适配版本 `0f84b193a0e01719cf3a2bb95ff5cfc9d0f3f4e8`。表中数字来自 **NPU 适配版**，不是未修改 main；运行时标记为 `2c23b68-npu-page16-tp-stop-worker-log`。
- vLLM-Omni main 固定为 `6d29fde51c08b2512eafa8778315c2eda745bebe`，NPU 适配预检版本 `0eb6f3c`；尚无完整服务 benchmark。
- 模型 `/dataset/models/drama-3`，HF revision `e0a9218aa48888270f06ef1ed9ea67622c0e17d2`；原始 block-FP8 expert 权重精确展开为 BF16。
- SGLang 使用物理 NPU 0/1、TP2；独立框架使用物理 6/7、层切分。共享主机存在并行实验，未完成独占设备同条件 AB/BA；不能将不同 cohort 的结果计算成正式加速比。
- 英文 SeedTTS 128 条固定 workload SHA256：`9bdd98e981c46225cb7a85362115f2e72e53ca94e17e38154f1e507d3917c2af`。每档 128 measured，warmup max(8, concurrency)。采样 temperature=1、top_p=0.9、top_k=30、max_frames=500；44.1 kHz、hop=2048，初始 chunk 2 帧、后续 25 帧、left context 25 帧。
- Throughput 使用完整 measured cohort wall time 与真实声学帧数；TTFA 为客户端首个音频数据到达，RTF 按音频时长归一化。请求完成、语音质量与进程退出分开判定。
- sweep 在 C64 崩溃后停止，完整 128 条 cohort 的 WER/SIM 尚未执行，不能将 C4 的质量结果套用于其它档位。

证据：`evidence/pr-publication/sglang-sweep-status.json`；完整实现与既有验证见 `README.md` 和 `evidence/campaign-20261017/README.md`。
