# PROGRESS

> 实时记录每个里程碑的状态、代码 tag、配套文章链接。完整计划见 [PLAN.md](PLAN.md)。

## 状态图例

- ⬜ 未开始
- 🟡 进行中
- ✅ 完成
- 🔁 升级中（已有版本，正在写更优实现）

## 核心里程碑（M1–M5）

| M | 状态 | Tag | 完成日期 | 文章 | 备注 |
| --- | --- | --- | --- | --- | --- |
| M1a Qwen3 数值对齐 | 🟡 | — | — | — | T1 RMSNorm ✅ (12/12 单测绿)；T2-T6 进行中 |
| M1b 单序列前向 | ⬜ | — | — | — | 最小 Engine + CLI 出字，Protocol 只钉必要契约 |
| M2 KV Cache | ⬜ | — | — | — | `ContiguousKVCache` |
| M3 Continuous Batching | ⬜ | — | — | — | `FCFSScheduler` + 三队列 |
| M4 PagedAttention (PyTorch) | ⬜ | — | — | — | `PagedKVCache`，伪版 |
| M5a API + SSE | ⬜ | — | — | — | OpenAI API + sampler 参数 |
| M5b Prefix + Reasoning | ⬜ | — | — | — | prefix cache + `reasoning_content` 分流 |
| M5c Benchmark + CI | ⬜ | — | — | — | 三栏对照 + GitHub Actions + v1.0 |

## 扩充里程碑（M6+）

| M | 状态 | Tag | 文章 | 备注 |
| --- | --- | --- | --- | --- |
| M6 MoE 教学版 (for-loop) | ⬜ | — | — | Registry 引入 |
| M7 Spec Decoding (n-gram) | ⬜ | — | — | `Drafter` Plugin |
| M8 Triton PagedAttention kernel | ⬜ | — | — | 需 NVIDIA GPU |
| M9 MoE grouped GEMM | ⬜ | — | — | |
| M10 EAGLE-1 spec | ⬜ | — | — | |
| M11 Long context (YaRN) | ⬜ | — | — | |
| M12 Chunked Prefill | ⬜ | — | — | |
| M13 VLM 教学版 | ⬜ | — | — | `inputs_embeds` 走通 |
| M14 VLM 工程化 | ⬜ | — | — | image hash prefix cache |

## M15+ 候选池

详见 [PLAN.md §4 M15+](PLAN.md)，按兴趣挑选开新 M。

## 日志

### 2026-06-06
- 仓库 `luhao2013/inferlite` 创建（MIT，公开）
- 完整 PLAN 落地（含 4 层抽象 / L0–L3 四层验证 / Benchmark 三件套 / 14 个里程碑）
- M1 收窄为 M1a（数值对齐）+ M1b（Engine/CLI 出字），避免首阶段 DoD 过载

### 2026-06-07
- **T1 RMSNorm 完成** (commit `d36b5da`)
  - `inferlite/model/layers.py::RMSNorm` 与 `transformers.Qwen3RMSNorm` 数值对齐
  - `tests/unit/test_rmsnorm.py`：3 shape × 3 dtype + 3 invariant = 12 单测全绿
  - 教学级注释加在实现与测试两处
- **CI / pre-commit 上线** (commit `d36b5da`)
  - `.github/workflows/tests.yml`: ubuntu + macos 双平台 py3.12
  - `.pre-commit-config.yaml`: 行尾/yaml/toml/large-file + ruff lint/format
- **地基补完善** (本 commit)
  - `scripts/setup.sh` 加包骨架 + pre-commit hook 自动注册
  - `RMSNorm.variance_eps` 重命名为 `.eps`（与社区一致）
- **工具链**：make setup → make preflight (ModelScope) → uv run pytest → CI
