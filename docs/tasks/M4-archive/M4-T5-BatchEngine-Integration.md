# M4-T5 — BatchEngine Integration

> 把 T4 的 paged attention 能力接入 M3 continuous batching 主循环：用 `PagedKVCache` 替换 fixed-slot cache 路径，明确 request 生命周期、block admission、allocate/append/free 的责任边界。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T5 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ⬜ pending |
| 前置 | M4-T4 PagedAttention PyTorch ✅ |
| 后续 | M4-T6 — E2E Correctness & Benchmark |
| 估时 | 4～6h |
| 核心文件 | `inferlite/engine/paged_core.py`（优先新建） |
| 可能涉及 | `inferlite/engine/batch_core.py`、`inferlite/model/model.py`、`inferlite/engine/protocol.py` |
| 测试文件 | `tests/unit/test_paged_batch_engine.py` 或 `tests/e2e/test_paged_batch_generate.py` |

## 范围冻结

T5 是 **engine 集成任务**。T4 只证明单层 attention 能用 paged KV；T5 负责把这个能力接入真实 batch generation，让请求从 waiting 到 running 再到 finished 的生命周期中正确分配和释放物理 blocks。

### 明确做

- 新增 `batch_generate_paged()` 或等价 paged batch generation 入口。
- 保持 M3 fixed-slot `batch_generate` 路径可作为 oracle，不删除旧实现。
- prefill 阶段为 admitted request 调 `PagedKVCache.allocate_request(request_id, prompt_len)`。
- 将 `paged_kv_cache`、`layer_idx`、`request_ids`、`paged_is_prefill` 透传到每层 attention。
- decode 阶段每轮对 running requests 先调 `PagedKVCache.append_token(request_id)`，再走 paged attention decode。
- finished / EOS / max_new_tokens 时调用 `PagedKVCache.free_request(request_id)`。
- 做最小 block admission：`cache.can_allocate(prompt_len)` 为真才 admit；不足则留在 waiting。
- 证明 paged path 输出 token 语义对齐 M3 fixed-slot path。

### 明确不做

- 不做 benchmark 结果归档（T6）。
- 不更新 README/PROGRESS/tag（T8）。
- 不做 prefix sharing、hash、LRU、CoW（M5）。
- 不做复杂抢占、swap、recompute；block 不足时 waiting 排队即可。
- 不优化 PyTorch gather 性能；性能分析留 T6，kernel 化留 M9。
- 不把 M3 fixed-slot 路径删除或强行合并。

## 背景

M3 的 continuous batching 已经解决「多个请求如何在 decode iteration 边界重新组 batch」，但 KV 存储是 fixed-slot：

```text
BatchedKVCache.k: [max_num_slots, n_kv_heads, max_seq_len, head_dim]
```

T1～T4 已经完成 paged memory 与 attention 单层能力：

```text
BlockPool       -> 管物理 block 分配/释放/ref_count
BlockTable      -> 管 request 逻辑位置到物理 block 的映射
PagedKVCache    -> 管真实 K/V tensor 与 batch scatter/gather
PagedAttention  -> attention 层消费 PagedKVCache
```

T5 要把这些串成一条 engine 路径：

```text
waiting requests
  -> block admission
  -> allocate_request
  -> paged prefill
  -> running decode loop
  -> append_token + paged decode
  -> finished -> free_request
```

## 推荐入口设计

优先新建：

```python
def batch_generate_paged(...): ...
```

理由：

- M3 fixed-slot 路径已经稳定，是 T5/T6 的 oracle。
- 新入口让 paged 集成变量更少，便于回滚和对比。
- 等 T6 验证完成后，再考虑是否抽象公共 engine loop。

不建议 T5 一开始就把 fixed-slot 和 paged 硬合进一个大函数。若要抽公共 helper，只抽与 cache 无关的纯逻辑，例如 request finished 判断、metrics 汇总等。

## 生命周期合同

### 1. Admission

waiting request 只有在 cache 空间足够时才进入 running：

```python
if paged_cache.can_allocate(prompt_len):
    admit(request)
else:
    keep_waiting(request)
```

T5 的最小 admission 策略：

- FCFS。
- 不抢占 running。
- 不拆 prompt。
- 不 chunked prefill。
- 不为了后面的短请求跳过队首长请求，除非 M3 已有策略要求这样做。

### 2. Prefill

admitted request 进入 running 前或进入 running 当轮：

```python
paged_cache.allocate_request(request_id, prompt_len)
model(..., paged_kv_cache=paged_cache, request_ids=[request_id], paged_is_prefill=True)
```

边界：

- `allocate_request` 只调用一次。
- T5 负责生成稳定唯一的 `request_id`。
- prefill 可先按单请求逐个做，M4 不要求 chunked/batched prefill。
- 如果 prefill 失败，必须释放已分配 block 或保证异常前未注册成功；不得泄漏。

### 3. Decode

每个 decode iteration：

```python
request_ids = [req.request_id for req in running]
for request_id in request_ids:
    paged_cache.append_token(request_id)
model(..., paged_kv_cache=paged_cache, request_ids=request_ids, paged_is_prefill=False)
```

边界：

- `append_token` 必须在写当前 token K/V 前调用。
- `request_ids` 顺序必须与 batch input 行顺序完全一致。
- 任一 append 因 block 不足失败时，T5 暂不做抢占；可保持请求 waiting/running 状态并抛清晰异常，或在 admission 阶段预留 decode 容量。具体实现策略必须写入完成总结。

### 4. Finish / Free

请求 finished 条件与 M3 一致：

- 生成 EOS。
- 达到 `max_new_tokens`。
- 上层取消（若已有接口支持）。

finished 后必须：

```python
paged_cache.free_request(request_id)
```

释放是 T5 的硬门禁。测试必须覆盖：finished 后 `num_free_blocks` 恢复，waiting 请求能继续进入。

## 接口透传边界

T4 没有打通完整 model 调用链，T5 需要做最小透传：

```text
batch_generate_paged
  -> Qwen3ForCausalLM / model wrapper
    -> Qwen3Model
      -> DecoderLayer(layer_idx=i)
        -> GQAAttention.forward(
             paged_kv_cache=paged_cache,
             layer_idx=i,
             request_ids=request_ids,
             paged_is_prefill=...
           )
```

实现时必须保证：

- `layer_idx` 是 decoder layer 的真实下标，不是 batch 下标。
- `request_ids` 顺序与 `input_ids` batch 行顺序一致。
- prefill/decode 标志显式传递，不靠 token 维度猜。
- 不破坏 M1/M2/M3 现有模型调用签名；新增参数都应有默认值。

如果透传需要改 `engine/protocol.py`，只增加 paged 所需的最小参数，并补注释说明这是 M4-T5 的调用链桥接。

## Block admission 策略

M4 最小策略：只检查 prompt prefill 所需 blocks：

```text
needed = ceil(prompt_len / block_size)
```

可选增强：为 decode 预留 1 个 block，减少刚 admit 后第一轮 decode 失败概率。但如果做预留，必须在任务卡完成总结里记录：

- 预留了多少；
- 是否影响并发；
- T6 benchmark 如何解释。

本任务默认不实现复杂 token budget，也不做 chunked prefill。

## 实现步骤

1. 阅读 M3 `batch_generate` / scheduler / request state，确认 fixed-slot 生命周期。
2. 新建 `engine/paged_core.py` 或最小扩展现有 batch engine。
3. 实现 paged cache 初始化与 request_id 生成/传递。
4. 实现 admission：`can_allocate(prompt_len)` 不满足则 waiting。
5. 实现 prefill：`allocate_request` + paged prefill 参数透传。
6. 实现 decode：`append_token` + paged decode 参数透传。
7. 实现 finish/free：EOS / max_new_tokens 后 `free_request`。
8. 保留/转发 metrics 基本字段，并新增 paged 相关最小指标（如 allocated/free blocks）。
9. 补 unit/e2e 测试，使用 M3 fixed-slot 作为 oracle。
10. 跑定向测试和 M3 回归。
11. 补齐实现与测试文件教学级注释，任务卡追加完成总结。

## 测试清单

| # | 测什么 | 预期 |
|---|---|---|
| 1 | M3 fixed path 回归 | 原 `batch_generate` 测试继续全绿 |
| 2 | 单请求 paged generate | 输出与 serial/M3 fixed-slot 等价 |
| 3 | 多请求变长 paged generate | token 级输出与 fixed-slot path 等价 |
| 4 | waiting > capacity | block 不足的请求留在 waiting，前序完成释放后继续进入 |
| 5 | finished 释放 block | EOS / max_new_tokens 后 `num_free_blocks` 恢复 |
| 6 | 跨 block decode | 长度正好跨 block 边界时输出正确 |
| 7 | request_id 顺序 | batch 行与 request_id 顺序错配会被测试捕获 |
| 8 | block 耗尽异常 | 无复杂抢占时抛出清晰错误或 admission 避免进入不可执行状态 |

### 测试命令

```bash
uv run pytest tests/unit/test_paged_batch_engine.py -q
uv run pytest tests/e2e/test_paged_batch_generate.py -q
uv run pytest tests/unit/test_batched_attention.py tests/unit/test_paged_attention.py tests/unit/test_paged_kv_cache.py -q
```

实际测试文件名可按项目现有结构调整，但必须覆盖上表语义。

## DoD

- [ ] 新增 paged batch generation 入口，M3 fixed-slot 入口不被删除。
- [ ] prefill 阶段正确调用 `allocate_request` 且只调用一次。
- [ ] decode 阶段每轮写 K/V 前正确调用 `append_token`。
- [ ] finished 请求正确调用 `free_request`，无 block 泄漏。
- [ ] 最小 block admission 生效；容量不足请求留在 waiting。
- [ ] paged path 多请求输出与 fixed-slot oracle token 级等价。
- [ ] 跨 block decode 测试通过。
- [ ] M3 fixed path 回归全绿。
- [ ] metrics 至少包含可解释的 block 使用信息。
- [ ] 实现与测试文件补齐教学级注释。
- [ ] 本任务卡追加完成总结与 commit 号。

## 坑（按概率排序）

1. **忘记 free_request**：短测可能不暴露，但长时间运行会 block 泄漏，waiting 永远进不来。
2. **decode 写 K/V 前忘记 append_token**：会覆盖旧 token 或写到错误 offset。
3. **request_ids 与 batch 行错配**：shape 全对但语义全错，必须用测试锁住。
4. **把 admission 和 prefix cache 混在一起**：M4 只按容量 admit，不做共享命中。
5. **为了复用 M3 过度改 batch_core**：容易破坏 stable oracle。优先新建 paged 入口。
6. **block 不足时状态半更新**：append/allocate 失败时不能让 request 状态和 cache 状态不一致。
7. **prefill/decode 靠 seq_len 猜**：T5 应显式传 `paged_is_prefill`。
8. **T5 顺手做 benchmark/attention 重构/文档收口**：这些留给 T6/T7/T8，避免任务失焦。

## 与后续任务的衔接

T6 基于 T5 的 paged engine 做 E2E 正确性和 benchmark。T5 完成时只需提供可运行、可测、可解释的 paged batch generation，不要求性能优于 M3。
