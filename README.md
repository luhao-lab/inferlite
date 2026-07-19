<div align="center">

# inferlite

**从零手写一个可读、可跑、可解释的 LLM 推理引擎**

覆盖 vLLM 核心思想 — KV cache · PagedAttention · Continuous Batching · Prefix Cache —
按里程碑驱动持续扩充（MoE / Spec Decoding / Triton / VLM …）

[![docs](https://img.shields.io/badge/📖_docs-online-26c6da?style=for-the-badge)](https://luhao-lab.github.io/inferlite/)
[![tests](https://github.com/luhao-lab/inferlite/actions/workflows/tests.yml/badge.svg)](https://github.com/luhao-lab/inferlite/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

[**🌐 在线文档站**](https://luhao-lab.github.io/inferlite/)  ·  [路线图](docs/plan/PLAN.md)  ·  [实时进度](docs/plan/PROGRESS.md)  ·  [当前作战 M3](docs/plan/PLAN.md)

</div>

---

## 项目定位

| 不变量 | 说明 |
| --- | --- |
| **代码全手敲** | 作者本人手写每一行 `inferlite/*.py`；Agent 仅辅助研究 / 计划 / Review / 文章 |
| **里程碑闭环** | 每个 M 完成 = ① 代码 push  ② 知乎文章发布  ③ PROGRESS 更新 |
| **学习 > 性能** | 优先可读性；性能优化作为后续里程碑慢慢加 |
| **Spec-driven** | 任务卡 7 字段（前置/边界/验收/风险/完成总结），见 [ADR-001](docs/kb/decisions.md) |

## 30 秒 Quick Start

```bash
git clone git@github.com:luhao-lab/inferlite.git
cd inferlite
make setup            # uv 装环境 + sanity check
make test             # 跑 123/123 单测，全绿
make preflight        # 一键拉 Qwen3-0.6B + 端到端跑一句话
```

更多命令见 [`make help`](docs/README.md) 或 `docs/README.md`。

## 路线图

```mermaid
flowchart LR
    subgraph CORE [核心 · demo 跑起来]
        direction TB
        M1[M1<br/>Qwen3 数值对齐<br/>+ 单序列前向]
        M2[M2<br/>KV Cache<br/>O n²→O n ]
        M3[M3<br/>Continuous<br/>Batching]
        M4[M4<br/>Paged<br/>Attention]
        M5[M5<br/>API + SSE +<br/>Prefix + CI]
        M1 --> M2 --> M3 --> M4 --> M5
    end
    subgraph EXT [扩充 · 无截止]
        direction TB
        M6[M6 MoE]
        M7[M7 Spec Decode]
        M8[M8 Triton kernel]
        M9[M9-M14 ...]
    end
    M5 --> M6
    M5 --> M7
    M5 --> M8
    M6 --> M9
```

完整 14 个里程碑见 [`docs/plan/PLAN.md`](docs/plan/PLAN.md)。

## 当前进度

- ✅ **M0** 仓库 + 计划 + 知识库脚手架
- ✅ **M1** Qwen3 数值对齐 + 单序列前向（tag: `m1/naive-forward`，2026-06-19）
  - 95 个单测全绿，Qwen3-0.6B e2e 与 transformers 精确对齐
- ✅ **M2** KV Cache（tag: `m2-complete`，2026-06-29）
  - T1~T5 全部完成，新增 28 个单测，M1/M2 端到端 bench 实测加速 7.36×
- ✅ **M3** Continuous Batching（tag: `m3/continuous-batching`，2026-07-19）
  - T1~T7 全部完成，fixed-slot continuous batching + metrics/benchmark，E2E 与 serial generate 等价
- ⬜ M4 PagedAttention · M5 API+SSE
- ⬜ M6+ MoE / Spec Decoding / Triton / VLM …

实时状态见 [`docs/plan/PROGRESS.md`](docs/plan/PROGRESS.md)。

## 文档导航

| 入口 | 内容 |
| --- | --- |
| 🌐 [**在线文档站**](https://luhao-lab.github.io/inferlite/) | 极客风深色主题 · 全文搜索 · mermaid 渲染 |
| 🗺️ [`docs/plan/`](docs/plan/) | PLAN（14 个 M）· PROGRESS · M1.md · M2.md |
| 📋 [`docs/tasks/`](docs/tasks/) | 任务卡（一卡一文件，七字段闭环） |
| 📚 [`docs/kb/`](docs/kb/) | knowledge / lessons / decisions / references |
| ⚙️ [`docs/README.md`](docs/README.md) | 环境 + 仓库结构速查 |
| 🤖 [`CLAUDE.md`](CLAUDE.md) | AI 协作约定（双轨制 · spec-driven） |

## 技术栈

- **Python** 3.12 + **PyTorch** 2.4+（当前 lock：2.12.0）
- **主模型**：Qwen3-0.6B（M1–M5 起步） · 通过 ModelScope 拉
- **Tokenizer**：复用 `transformers.AutoTokenizer`
- **数值对齐基准**：当前 `uv.lock` 锁定的 `transformers==5.10.2`
- **Server**：FastAPI + SSE（M5 引入）
- **硬件**：Mac MPS 主开发（M1–M7） · GPU 在 M5 benchmark / M8 Triton 必需
- **工具链**：uv · ruff · pytest · pre-commit · MkDocs Material

## 工作流（spec-driven · 5 个 slash commands）

```text
/plan <scope>      规划（M / T / 调整），含前置调研
/work <task>       开任务卡，含 knowledge gap 检查
/review <task>     review 已完成任务卡
/archive <id>      归档（沉淀 lessons + knowledge + summary）
/preflight         环境健康检查
```

详见 [`docs/kb/decisions.md`](docs/kb/decisions.md) ADR-001。

## License

MIT — 见 [LICENSE](LICENSE)。

---

<div align="center">
<sub>Built with ❤ + uv + PyTorch · Powered by <a href="https://luhao-lab.github.io/inferlite/">MkDocs Material</a></sub>
</div>
