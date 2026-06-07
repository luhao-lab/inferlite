# inferlite

> 从零手写一个**可读、可跑、可解释**的 LLM 推理引擎，覆盖 vLLM 的核心思想（KV cache / PagedAttention / Continuous Batching / Prefix Cache），按里程碑驱动持续扩充（MoE / Spec Decoding / Triton kernel / 量化 / VLM …）。

[![tests](https://github.com/luhao2013/inferlite/actions/workflows/tests.yml/badge.svg)](https://github.com/luhao2013/inferlite/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## 项目定位

- **代码全手敲**：作者本人手写每一行 `inferlite/*.py`，Agent 仅辅助研究 / 计划 / Review / 文章
- **里程碑闭环**：每个 M 完成 = ① 代码 push ② 知乎文章发布 ③ PROGRESS.md 更新
- **学习 > 性能**：优先可读性；性能优化作为后续里程碑慢慢加

## 路线图速览

| 阶段 | 里程碑 | 目标 |
| --- | --- | --- |
| **核心（demo 跑起来）** | M1a | Qwen3 数值对齐：算子 + logits allclose |
| | M1b | 单序列前向：Engine + CLI 能出字 |
| | M2 | KV Cache：从 O(n²) 到 O(n) |
| | M3 | Continuous Batching：调度器的诞生 |
| | M4 | PagedAttention（PyTorch 伪版） |
| | M5a | 采样 + OpenAI API + SSE |
| | M5b | 前缀缓存 + Reasoning 字段分流 |
| | M5c | Benchmark + CI + v1.0 |
| **扩充（无截止）** | M6 | MoE 教学版（for-loop） |
| | M7 | Speculative Decoding（n-gram） |
| | M8 | Triton PagedAttention kernel |
| | M9 / M10 | MoE grouped GEMM / EAGLE-1 |
| | M11 / M12 | Long context / Chunked Prefill |
| | M13 / M14 | VLM 教学版 / VLM 工程化 |
| | M15+ | 量化 / MLA / TP-PP / Hybrid SSM / Audio 输入 … |

完整计划见 [docs/1-plan/PLAN.md](docs/1-plan/PLAN.md)，实时进度见 [docs/1-plan/PROGRESS.md](docs/1-plan/PROGRESS.md)，M1 详细 brief 见 [docs/1-plan/M1.md](docs/1-plan/M1.md)，仓库目录与环境说明见 [docs/4-setup.md](docs/4-setup.md)。

## 环境

一键安装：

```bash
make setup
```

详细说明（含 uv / make 是什么、常用命令、踩坑预警）见 [docs/4-setup.md](docs/4-setup.md)。

知识点 / 教训 / 决策记录见：[knowledge.md](docs/3-kb/knowledge.md) · [lessons.md](docs/3-kb/lessons.md) · [decisions.md](docs/3-kb/decisions.md)。

## 技术栈

- Python 3.11 + PyTorch 2.4+（当前 lock：PyTorch 2.12.0）
- 主模型：**Qwen3-0.6B**（M1–M5 起步）
- 第二模型：M13 前再定，候选 **Qwen-VL / Llava / native multimodal 小模型**
- Tokenizer 复用 `transformers.AutoTokenizer`
- 数值对齐基准：当前 `uv.lock` 的 `transformers==5.10.2`
- Server：FastAPI + SSE
- 硬件：Mac MPS 主开发（M1–M7），GPU 在 M5 benchmark / M8 Triton 必需

## 状态

- [x] M0 仓库与计划落地
- [ ] M1a Qwen3 数值对齐
- [ ] M1b 单序列前向

## License

MIT
