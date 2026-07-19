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
| **M1** Qwen3 单序列推理 | ✅ | `m1/naive-forward` | 2026-06-19 | — | 95 个单测全绿，Qwen3-0.6B e2e 与 transformers 精确对齐 |
| **M2** KV Cache | ✅ | `m2/static-kv-cache` | 2026-06-29 | — | T1~T5 全部完成，+28 单测，端到端 bench 7.36×@T=512 |
| **M3** Continuous Batching | ✅ | `m3/continuous-batching` | 2026-07-19 | — | T1~T7 完成，fixed-slot continuous batching + metrics/benchmark，MPS 实测 batch throughput 慢于 serial，根因在 cache 路径 |
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
- 知识缺口归档到 `docs/knowledge/knowledge.md` 顶部"📊 索引摘要"段（首次会话即可看到）

### 2026-06-26 — M2 KV Cache 完成

- **T1 KVCache 数据结构** (`474c04f`)
  - `inferlite/model/kv_cache.py`：`LayerKVCache`（dataclass）+ `KVCache`（含 `from_config`/`reset`）
  - 静态预分配 28 层 × [B, n_kv, max_seq_len, head_dim] tensor，`cur_len` 是唯一事实源
  - `tests/unit/test_kv_cache.py`：from_config shape/dtype/device、reset、LayerKVCache 结构，共 7 个测试全绿

- **T2 Attention KV Cache 接口** (`8179b74`)
  - `GQAAttention.forward`：新增 `layer_kv_cache` + `cache_position` 参数
  - Prefill：写入 `k[:, :, :T_p, :]`；Decode：追加写入 + `k[:, :, :cur_len+1, :]` 做 Attention
  - `kv_cache=None` 完全兼容 M1 路径
  - `tests/unit/test_attention_kv.py`：prefill/decode 输出一致性、兼容路径，fp32 误差 < 1e-5

- **T3 Model Passthrough** (`1dd4c8d`)
  - `Qwen3Model.forward` + `Qwen3ForCausalLM.__call__` 新增 `kv_cache`/`position_ids` 透传
  - `position_embeddings` (cos/sin) 统一在 `Qwen3Model` 里算一次，传给各层
  - 全部 95 个 M1 单测继续通过，无回归

- **T4 Generate Loop 拆 Prefill/Decode** (`0e13a42`)
  - `generate()` 新增 `kv_cache` 参数；有 cache 时走 prefill（全量 prompt）+ decode loop
  - decode 每步 `position_ids = [[cur_len]]`（绝对位置），`cur_len` 在 generate 里维护
  - `engine/protocol.py`：`ModelProtocol.__call__` 新增 `position_ids`/`kv_cache` 参数
  - `tests/unit/test_generate_kv.py`：输出一致性、长度、EOS 停止、绝对位置、reset 复用，共 5 个测试

- **T5 CLI device/dtype/max-seq-len** (`a0a2004`)
  - `--device auto/cpu/mps/cuda`、`--dtype auto/bf16/fp16/fp32`、`--max-seq-len`
  - `resolve_device_dtype()`：auto 优先级 mps > cuda > cpu，bf16 on gpu
  - `model.to(device, dtype=dtype)` + `KVCache.from_config(..., device, dtype)` 接入
  - `tests/unit/test_cli.py`：10 个用例覆盖新参数解析、resolve 逻辑、kv_cache wiring

### 2026-06-07（晚）— 整体规划体检 & R2 微调
- M1 任务编号统一：去掉 `T0'/T0p` 撇号 → **T0** ModelConfig；其他 T1-T11 不动
- M1·P1 / M1·P2 替代 M1a / M1b（保留 Phase 概念，不再算两个独立里程碑）
- M5 合并：M5a/M5b/M5c 改为 M5 单一里程碑 + 三个内部 Phase（与 M1 同思路）
- M11 ↔ M12 顺序调整：Chunked Prefill 提前到 M11（长上下文前置依赖），Long context (YaRN) 后移到 M12
- M1.md §4 任务总表：新增 `前置` 列 + `[P]` 并行标记列 → 一眼看出 T1/T2/T3 三线可并开
- `scripts/doctor.sh` + `make doctor`：跨文档一致性自检（任务卡 ↔ M1.md ↔ PROGRESS ↔ README）
- 知识缺口归档到 `docs/knowledge/knowledge.md` 顶部"📊 索引摘要"段（首次会话即可看到）

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
