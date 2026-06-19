# PROGRESS

> 实时记录每个里程碑的状态、代码 tag、配套文章链接。完整计划见 [PLAN.md](PLAN.md)。

## 状态图例

- ⬜ 未开始
- 🟡 进行中
- ✅ 完成
- 🔁 升级中（已有版本，正在写更优实现）

## 核心里程碑（M1–M5）

> 状态用 **整体里程碑** 维度记录；M1 / M5 内部 Phase 进度看对应章节文档（M1.md、PLAN §3 M5）。

| M | 状态 | Tag | 完成日期 | 文章 | 备注 |
| --- | --- | --- | --- | --- | --- |
| **M1** Qwen3 单序列推理 | 🟡 | — | — | — | P1 数值对齐（T0/T1/T2 ✅, T3-T8 ⬜）+ P2 出字（T9-T11 ⬜） |
| **M2** KV Cache | ⬜ | — | — | — | `ContiguousKVCache` |
| **M3** Continuous Batching | ⬜ | — | — | — | `FCFSScheduler` + 三队列 |
| **M4** PagedAttention (PyTorch) | ⬜ | — | — | — | `PagedKVCache`，伪版 |
| **M5** 服务化收口 v1 | ⬜ | `v1.0` | — | — | P1 API+SSE / P2 Prefix+Reasoning / P3 Benchmark+CI |

## 扩充里程碑（M6+）

| M | 状态 | Tag | 文章 | 备注 |
| --- | --- | --- | --- | --- |
| M6 MoE 教学版 (for-loop) | ⬜ | — | — | Registry 引入 |
| M7 Spec Decoding (n-gram) | ⬜ | — | — | `Drafter` Plugin |
| M8 Triton PagedAttention kernel | ⬜ | — | — | 需 NVIDIA GPU |
| M9 MoE grouped GEMM | ⬜ | — | — | |
| M10 EAGLE-1 spec | ⬜ | — | — | |
| **M11 Chunked Prefill** | ⬜ | — | — | 长上下文先要"喂得进"，调度层前置 |
| **M12 Long context (YaRN)** | ⬜ | — | — | RoPE 频率重映射，依赖 M11 |
| M13 VLM 教学版 | ⬜ | — | — | `inputs_embeds` 走通 |
| M14 VLM 工程化 | ⬜ | — | — | image hash prefix cache |

## M15+ 候选池

详见 [PLAN.md §4 M15+](PLAN.md#milestones-extension)，按兴趣挑选开新 M。

### 2026-06-09
- **T0 ModelConfig 完成**
  - `inferlite/config.py::ModelConfig`：11 个 Qwen3-0.6B 核心超参，`frozen=True` 只读合同
  - `from_json()`：白名单过滤 HF config.json，`head_dim` 缺失兼容兜底，`rope_theta` cast float
  - `qwen3_0_6b()`：硬编码 0.6B ground truth，单测不依赖磁盘缓存
  - `tests/unit/test_config.py`：factory / JSON round-trip / head_dim fallback / frozen / GQA validation 共 5 测试
  - 验证：`uv run pytest tests/unit/test_config.py -q` 5/5 绿；`make test` 17/17 绿；`make doctor` 9/9 绿
  - 复盘：补 Qwen3-0.6B 架构精读、Python dataclass / Factory pattern 知识卡；新增 L4 head_dim 独立超参教训
- **T2 SwiGLUMLP 完成**
  - `inferlite/model/layers.py::SwiGLUMLP`：`gate_proj / up_proj / down_proj` 三个 `bias=False` Linear
  - forward 与 `transformers.Qwen3MLP` 对齐：`down_proj(F.silu(gate_proj(x)) * up_proj(x))`
  - `tests/unit/test_mlp.py`：dtype 对齐 / bias / shape invariant / 3 Linear / 0.6B 权重形状，共 10 个 case
  - 验证：`uv run pytest tests/unit/test_mlp.py -q` 10/10 绿

## 日志

### 2026-06-06
- 仓库 `luhao-lab/inferlite` 创建（MIT，公开）
- 完整 PLAN 落地（含 4 层抽象 / L0–L3 四层验证 / Benchmark 三件套 / 14 个里程碑）
- M1 收窄为 M1·P1（数值对齐）+ M1·P2（Engine/CLI 出字），避免首阶段 DoD 过载

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

### 2026-06-07（晚）— 整体规划体检 & R2 微调
- M1 任务编号统一：去掉 `T0'/T0p` 撇号 → **T0** ModelConfig；其他 T1-T11 不动
- M1·P1 / M1·P2 替代 M1a / M1b（保留 Phase 概念，不再算两个独立里程碑）
- M5 合并：M5a/M5b/M5c 改为 M5 单一里程碑 + 三个内部 Phase（与 M1 同思路）
- M11 ↔ M12 顺序调整：Chunked Prefill 提前到 M11（长上下文前置依赖），Long context (YaRN) 后移到 M12
- M1.md §4 任务总表：新增 `前置` 列 + `[P]` 并行标记列 → 一眼看出 T1/T2/T3 三线可并开
- `scripts/doctor.sh` + `make doctor`：跨文档一致性自检（任务卡 ↔ M1.md ↔ PROGRESS ↔ README）
- 知识缺口归档到 `docs/3-kb/knowledge.md` 顶部"📊 索引摘要"段（首次会话即可看到）

## 日志

### 2026-06-06
- 仓库 `luhao-lab/inferlite` 创建（MIT，公开）
- 完整 PLAN 落地（含 4 层抽象 / L0–L3 四层验证 / Benchmark 三件套 / 14 个里程碑）
- M1 收窄为 M1·P1（数值对齐）+ M1·P2（Engine/CLI 出字），避免首阶段 DoD 过载

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
