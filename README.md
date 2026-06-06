# inferlite

> 从零手写一个**可读、可跑、可解释**的 LLM 推理引擎，覆盖 vLLM 的核心思想（KV cache / PagedAttention / Continuous Batching / Prefix Cache），按里程碑驱动持续扩充（MoE / Spec Decoding / Triton kernel / 量化 / VLM …）。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目定位

- **代码全手敲**：作者本人手写每一行 `inferlite/*.py`，Agent 仅辅助研究 / 计划 / Review / 文章
- **里程碑闭环**：每个 M 完成 = ① 代码 push ② 知乎文章发布 ③ PROGRESS.md 更新
- **学习 > 性能**：优先可读性；性能优化作为后续里程碑慢慢加

## 路线图速览

| 阶段 | 里程碑 | 目标 |
| --- | --- | --- |
| **核心（demo 跑起来）** | M1 | 单序列前向：模型能出字 |
| | M2 | KV Cache：从 O(n²) 到 O(n) |
| | M3 | Continuous Batching：调度器的诞生 |
| | M4 | PagedAttention（PyTorch 伪版） |
| | M5 | 采样 + 前缀缓存 + OpenAI API + Reasoning + Benchmark |
| **扩充（无截止）** | M6 | MoE 教学版（for-loop） |
| | M7 | Speculative Decoding（n-gram） |
| | M8 | Triton PagedAttention kernel |
| | M9 / M10 | MoE grouped GEMM / EAGLE-1 |
| | M11 / M12 | Long context / Chunked Prefill |
| | M13 / M14 | VLM 教学版 / VLM 工程化 |
| | M15+ | 量化 / MLA / TP-PP / Hybrid SSM / Audio 输入 … |

完整计划见 [docs/PLAN.md](docs/PLAN.md)，实时进度见 [docs/PROGRESS.md](docs/PROGRESS.md)。

## 技术栈

- Python 3.11 + PyTorch 2.4+
- 主模型：**Qwen3-0.6B**（M1–M5 起步）
- 第二模型：**Qwen3.5-0.8B**（M13 多模态起点，native multimodal）
- 备用：Llama-3.2-1B
- Tokenizer 复用 `transformers.AutoTokenizer`
- Server：FastAPI + SSE
- 硬件：Mac MPS 主开发（M1–M7），GPU 在 M5 benchmark / M8 Triton 必需

## 状态

- [x] M0 仓库与计划落地
- [ ] M1 单序列前向

## License

MIT
