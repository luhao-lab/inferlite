# M4-T8 — Docs & Tag

> 在 T4～T7 代码、E2E、benchmark 和 attention/backend 边界整理都完成后，做 M4 里程碑闭环：更新 README / PROGRESS / M4 plan / knowledge，检查任务卡完成总结和最终门禁，准备 annotated tag `m4/paged-attention`。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T8 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ⬜ pending |
| 前置 | M4-T7 Attention Backend Refactor ✅ |
| 后续 | M5 — Prefix Caching |
| 估时 | 2～3h |
| 核心文件 | `README.md`、`docs/plan/PROGRESS.md`、`docs/plan/M4.md`、`docs/knowledge/` |
| 产物 | M4 文档闭环、最终验证记录、annotated tag |

## 范围冻结

T8 是 **文档与里程碑收口任务**，不再实现新能力。它的职责是把 T4～T7 已经完成并验证过的事实沉淀到项目文档、进度记录和 tag 中。

### 明确做

- 更新 README 当前进度与 M4 成果摘要。
- 更新 `docs/plan/PROGRESS.md`：M4 状态、完成日期、验证摘要、tag。
- 更新 `docs/plan/M4.md`：完成定义勾选、实际实现边界、ADR 是否变化。
- 更新/补齐 `docs/knowledge/m4-paged-attention.md` 或相关 knowledge 文档。
- 记录 T7 attention/backend 边界整理后的最终结构。
- 检查 M4-T1～T8 每张任务卡是否有完成总结和 commit 号。
- 记录 T6 benchmark 结果路径和关键结论。
- 跑最终门禁：ruff / format、定向测试、全量回归。
- 准备 annotated tag `m4/paged-attention`。

### 明确不做

- 不修核心实现 bug；发现 bug 回退到对应 T4/T5/T6/T7。
- 不新增 Prefix Cache 设计细节；最多说明 M4 为 M5 准备了 BlockPool/ref_count/block table 和 attention backend 边界。
- 不把未完成项口头标成完成。
- 不为了 tag 跳过测试、注释、benchmark 或任务卡总结。
- 不改 benchmark 结果；只引用和解释 T6 已归档结果。

## 文档更新清单

### 1. README

更新「当前进度」：

- M4 从 🟡 改为 ✅（只有所有 DoD 达成后）。
- 写明 tag：`m4/paged-attention`。
- 摘要说明：
  - BlockPool / BlockTable / PagedKVCache；
  - PyTorch gather 伪版 PagedAttention；
  - paged batch generation；
  - fixed-slot vs paged E2E 等价；
  - attention/cache backend 边界整理；
  - benchmark 结论。

不要在 README 写过长实现细节，细节放 knowledge。

### 2. `docs/plan/PROGRESS.md`

更新 M4 表格行：

- 状态：✅。
- Tag：`m4/paged-attention`。
- 完成日期：实际完成日期。
- 备注：T1～T8 完成、验证摘要、benchmark 路径。

追加日志条目，记录：

- T4 attention 层闭环；
- T5 paged engine 集成；
- T6 E2E/benchmark；
- T7 attention/backend 边界整理；
- T8 docs/tag。

### 3. `docs/plan/M4.md`

更新：

- 顶部状态从 🟡 改为 ✅。
- 任务表 T1～T8 状态。
- 完成定义 checklist。
- ADR 是否有变更：
  - M4 不含 CoW；
  - PyTorch gather 伪版；
  - T4 attention-only，T5 engine 集成；
  - T7 将 cache 分支收敛到 context/backend 边界；
  - Benchmark 只解释机制收益，不以吞吐超过 M3 为目标。
- 如果 T5/T6/T7 实现有与原计划不同的边界，写入「实际取舍」。

### 4. Knowledge 文档

更新或补齐 `docs/knowledge/m4-paged-attention.md`，至少包含：

```text
BlockPool -> BlockTable -> PagedKVCache -> PagedAttention -> PagedEngine -> Attention Backend Boundary
```

要解释：

- 逻辑 block 与物理 block 的映射。
- `slot = block_id * block_size + offset`。
- T3 batch scatter / gather 数据流。
- T4 为什么需要 valid_lens 同时清零 K/V 和 mask scores。
- T5 request 生命周期：allocate / append / free。
- T6 fixed-slot vs paged 内存/碎片对比。
- T7 为什么整理 `GQAAttention.forward` 与 cache/backend 边界。
- PyTorch gather 伪版与 M9 kernel backend 的关系。
- M5 Prefix Cache 将复用哪些基础设施和边界。

### 5. Lessons

如 T4～T7 出现可复用坑，更新 `docs/knowledge/lessons.md`。候选：

- `valid_lens` mask 与 K/V 清零必须成对出现。
- request_id 顺序是 batch cache 路径的隐式不变量。
- block 生命周期测试必须覆盖 free 后 waiting admission。
- attention 主流程不应继续堆叠 cache 策略分支；新策略应进入 context/backend 层。

没有新教训则不强行添加。

### 6. 任务卡完成总结

检查 M4-T1～T8：

- 每张卡末尾有 `## 完成总结`。
- 总结包含：
  - 最终能力边界；
  - 关键设计结论；
  - 已知限制；
  - 验证命令与结果；
  - commit 号。

T8 不伪造 commit。若某任务未提交，标明「待提交」并不得创建最终 tag。

## 最终门禁

T8 创建 tag 前必须完成：

```bash
uv run ruff check inferlite tests scripts
uv run ruff format --check inferlite tests scripts
uv run pytest tests/unit/test_paged_attention.py -q
uv run pytest tests/unit/test_paged_kv_cache.py -q
uv run pytest tests/e2e/test_paged_batch_generate.py -q
uv run pytest tests/ -q
```

实际命令可按项目文件名调整；完成总结必须记录真实执行过的命令和结果。

如果全量测试因环境限制无法运行，必须记录：

- 哪条命令无法运行；
- 原因；
- 已完成的替代验证；
- 是否允许暂缓 tag。

默认规则：**全量回归未通过或未跑清楚，不创建最终 tag。**

## Tag 流程

前置：

- 工作区干净或只剩预期文档变更。
- 所有 M4 代码/测试/文档已 commit。
- PROGRESS/README/M4/knowledge 已更新。
- benchmark 结果已归档。
- 最终门禁通过。

创建 annotated tag：

```bash
git tag -a m4/paged-attention -m "M4 PagedAttention"
```

是否 push tag 由用户确认；任务卡只记录准备和本地创建步骤，不擅自推送远端。

## 实现步骤

1. 汇总 T4～T7 完成总结、验证命令、commit 和 benchmark 结果路径。
2. 更新 README 当前进度。
3. 更新 `docs/plan/PROGRESS.md`。
4. 更新 `docs/plan/M4.md`。
5. 更新 knowledge / lessons。
6. 检查 M4-T1～T8 任务卡完成总结完整性。
7. 跑最终门禁。
8. 确认工作区与 commit 状态。
9. 创建 annotated tag。
10. T8 任务卡追加完成总结与 tag 信息。

## DoD

- [ ] README M4 状态、成果和 tag 更新。
- [ ] `docs/plan/PROGRESS.md` M4 状态为 ✅，含完成日期和验证摘要。
- [ ] `docs/plan/M4.md` 完成定义勾选，实际取舍记录清楚。
- [ ] knowledge 文档解释完整 M4 数据流、attention/backend 边界和 benchmark 结论。
- [ ] lessons 如有新增教训已更新。
- [ ] M4-T1～T8 每张任务卡都有完成总结和 commit 号。
- [ ] T6 benchmark 结果已归档并被文档引用。
- [ ] T7 attention/backend 重构结果已被文档引用。
- [ ] ruff / format / 定向测试 / 全量回归通过或限制记录清楚。
- [ ] annotated tag `m4/paged-attention` 已创建。
- [ ] 未引入 M5 Prefix Cache 等新能力。

## 坑（按概率排序）

1. **发现 bug 但在 T8 顺手修**：T8 只能收口，bug 回退到对应任务，避免文档任务混入实现变更。
2. **全量测试没跑就 tag**：违反 README 周完成门禁。
3. **任务卡完成总结缺 commit 号**：后续复盘无法追溯。
4. **README / PROGRESS / M4.md 状态不一致**：必须统一 M4 状态、tag、完成日期。
5. **benchmark 结论写得像性能胜利**：M4 可能更慢，重点是内存机制和碎片降低。
6. **过度展开 M5 设计**：T8 只说明 M4 为 M5 准备了基础，不提前写 Prefix Cache 实现细节。
7. **漏记 T7 架构变化**：如果 attention/backend 边界已整理，knowledge 必须同步说明。
8. **tag 前工作区不干净**：必须确认 commit 状态，避免 tag 指向半成品。
9. **未记录环境限制**：若某些命令无法跑，必须明确原因和风险。

## 与 M5 的衔接

M4 完成后，M5 Prefix Caching 的前置基础应包括：

- `BlockPool` 的 ref_count 基础；
- per-request `BlockTable`；
- `PagedKVCache` 的 block table 读写；
- paged engine 的 request 生命周期；
- attention/cache context 或 backend 边界；
- benchmark 对 block 使用和碎片的度量方法。

T8 只记录这些前置已具备，不实现 M5。
