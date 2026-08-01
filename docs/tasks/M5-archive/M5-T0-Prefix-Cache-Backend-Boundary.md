# M5-T0 — Prefix Cache 接入边界更新

> 在 M5 开始实现 hash/LRU/CoW 之前，先冻结 Prefix Cache 与 M4 attention/backend 边界的关系：M5 不再把新分支直接塞进 `GQAAttention.forward`，而是通过 M4-T7 形成的 cache context/backend 边界接入。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M5-T0 |
| 里程碑 | M5 — Prefix Caching |
| 状态 | ⬜ pending |
| 前置 | M4-T7 Attention Backend Refactor ✅、M4-T8 Docs & Tag ✅ |
| 后续 | M5-T1 BlockPool Hash + LRU |
| 估时 | 1～2h |
| 核心文件 | `docs/plan/M5.md`、后续 M5 任务卡 |
| 产物 | M5 接入边界冻结，避免 Prefix Cache 污染 attention 主流程 |

## 背景

M5 会引入：

```text
hash-based prefix caching
LRU eviction
partial hit CoW
shared block ref_count
```

这些能力都和 KV block 生命周期相关，但不属于模型 attention 数学本身。如果直接在 `GQAAttention.forward` 中新增 prefix cache 分支，会抵消 M4-T7 对 attention 主流程的整理。

## 范围冻结

### 明确做

- 明确 M5 Prefix Cache 通过 cache manager/backend/context 接入。
- 明确 hash/LRU/CoW/ref_count 逻辑位于 `BlockPool` / `PagedKVCache` / cache backend 层。
- 明确 `GQAAttention.forward` 只消费统一后的 KV 结果与 mask 元信息。
- 更新 M5 任务卡，要求每张实现卡不得新增 attention 主流程分支。
- 在 M5 E2E 中增加回归：M4-T7 后的 M1/M2/M3/M4 attention 路径不回归。

### 明确不做

- 不实现 prefix cache 本身；具体 hash/LRU/CoW 留给 M5-T1～T3。
- 不改 M4 已完成的 PagedAttention 语义。
- 不引入 M9 kernel metadata。
- 不把 vLLM `AttentionMetadata` 完整照搬进 M5。

## 接入原则

M5 后的职责分层：

```text
GQAAttention.forward
  只做模型 attention 主流程

AttentionCacheContext / backend
  描述当前 batch/cache 模式，返回 k/v + mask metadata

PagedKVCache
  管理 request -> block table -> physical block 的读写

BlockPool
  管理 ref_count / hash / LRU / eviction
```

Prefix Cache 命中与否应体现在：

- request 的 block table 如何构造；
- block 的 ref_count 如何变化；
- prefill 需要跳过哪些已命中 block；
- cache backend 返回的 KV 结果是否仍符合 attention core 合同。

而不是体现在：

```python
if prefix_cache_hit:
    ...  # 写在 GQAAttention.forward 里
```

## 对后续 M5 任务卡的要求

- M5-T1 BlockPool Hash + LRU：只改 block 元数据与分配策略。
- M5-T2 Prefix Cache Allocate：只改 request admission / block table 构造。
- M5-T3 Partial Hit CoW：只改 block table 替换与 KV block copy。
- M5-T4 Hash Registration：只改满 block 的 hash 注册。
- M5-T5 Beam Search：上层独立请求，依赖 Prefix Cache 自动共享。
- M5-T6 E2E + Benchmark：验证命中、跳过 prefill、TTFT/吞吐收益。
- M5-T7 Docs + Tag：记录 Prefix Cache 如何复用 M4-T7 backend 边界。

## DoD

- [ ] M5 plan/任务卡明确 Prefix Cache 不污染 `GQAAttention.forward`。
- [ ] M5 各任务的职责落在 BlockPool/PagedKVCache/cache backend/engine 层。
- [ ] M5 测试计划包含 M4 attention/backend 回归。
- [ ] 文档说明 M5 与 M4-T7 的衔接。
