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
| 可能涉及 | `inferlite/engine/batch_core.py`（只读参考）、`inferlite/model/qwen3.py`（加可选参数）、`inferlite/engine/protocol.py`（扩展 Protocol） |
| 测试文件 | `tests/unit/test_paged_batch_engine.py` 或 `tests/e2e/test_paged_batch_generate.py` |

## 范围冻结

T5 是 **engine 集成任务**。T4 只证明单层 attention 能用 paged KV；T5 负责把这个能力接入真实 batch generation，让请求从 waiting 到 running 再到 finished 的生命周期中正确分配和释放物理 blocks。

### 明确做

- 新增 `batch_generate_paged()` 或等价 paged batch generation 入口。
- 保持 M3 fixed-slot `batch_generate` 路径可作为 oracle，不删除旧实现。
- prefill 阶段为 admitted request 调 `PagedKVCache.allocate_request(request_id, prompt_len)`。多个 admitted 请求合并为一次 batched prefill 前向（pad 到最长 prompt 长度，T4 attention 层已支持变长 mask）。
- 将 `paged_kv_cache`、`request_ids`、`is_prefill` 透传到每层 attention（`layer_idx` 由 `Qwen3Model` 的 `enumerate` 自动生成，不需要顶层传入）。
- decode 阶段每轮对 running requests 先调 `PagedKVCache.append_token(request_id)`，再走 paged attention decode。
- finished / EOS / max_new_tokens 时调用 `PagedKVCache.free_request(request_id)`。
- 做 block-aware admission：`cache.can_allocate(prompt_len)` 为真才 admit；不足则留在 waiting。decode 按需分配新 block，block 不足时优雅结束请求。admission 在 `paged_core.py` 内 inline 实现，不新建 scheduler 文件。
- 证明 paged path 输出 token 语义对齐 M3 fixed-slot path（最小闭环验证，全面 token 级等价留 T6）。

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

admitted request 进入 running 前或进入 running 当轮，合并为一次 batched prefill：

```python
if admitted:
    max_prompt_len = max(req.prompt_ids.shape[1] for req in admitted)
    batch_input_ids = pad_and_cat([req.prompt_ids for req in admitted], max_prompt_len)
    batch_position_ids = make_position_ids([req.prompt_ids.shape[1] for req in admitted], max_prompt_len)
    batch_request_ids = [req.request_id for req in admitted]

    for req in admitted:
        paged_cache.allocate_request(req.request_id, req.prompt_ids.shape[1])

    logits = model(
        batch_input_ids,
        position_ids=batch_position_ids,
        paged_kv_cache=paged_cache,
        request_ids=batch_request_ids,
        is_prefill=True,
    )
```

边界：

- `allocate_request` 每个请求只调用一次。
- T5 负责生成稳定唯一的 `request_id`。
- prefill 采用 batched 前向：多个 admitted 请求 pad 到最长 prompt 长度，一次前向完成。T4 attention 层的 `_build_valid_lens_mask` 已支持变长 mask，padding 部分不会写入 KV cache。
- 如果 prefill 失败，必须释放已分配 block 或保证异常前未注册成功；不得泄漏。

### 3. Decode

每个 decode iteration：

```python
request_ids = [req.request_id for req in running]
# 每个请求的 decode position = 当前 seq_len（从 block_table 获取）
positions = torch.tensor([paged_cache.block_tables[rid].seq_len for rid in request_ids])
position_ids = positions.unsqueeze(1)  # [B, 1]
for request_id in request_ids:
    paged_cache.append_token(request_id)
model(..., paged_kv_cache=paged_cache, request_ids=request_ids, is_prefill=False)
```

边界：

- `append_token` 必须在写当前 token K/V 前调用。
- `request_ids` 顺序必须与 batch input 行顺序完全一致。
- 任一 append 因 block 不足失败时，T5 暂不做抢占，优雅结束该请求（保留已生成 token，mark_finished）。高并发场景的动态分配 + preemption 留后续里程碑对齐 vLLM V1 时处理。

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

## 接口透传方案

T4 没有打通完整 model 调用链，T5 需要做最小透传。经 vLLM V1 架构调研后，确定以下方案：

### 方案选择：直接加可选参数（方案 A）

给 `Qwen3ForCausalLM` / `Qwen3Model` / `DecoderLayer` 各加 3 个可选参数（默认 None/False），直接透传到 `GQAAttention`。T7 会用 `AttentionCacheContext` 收敛这些参数，T5 先用最简单的方式打通。

**不采用** vLLM V1 的 forward context / bind 机制——需要 ForwardContext 等基础设施，对教学项目过重。

### 模型签名变更

```python
# Qwen3ForCausalLM.forward 新增 3 个可选参数：
def forward(self, input_ids, ...,
            paged_kv_cache: PagedKVCache | None = None,
            request_ids: list[str] | None = None,
            is_prefill: bool = False) -> torch.Tensor

# Qwen3Model.forward 同样新增 3 个：
def forward(self, input_ids, ...,
            paged_kv_cache: PagedKVCache | None = None,
            request_ids: list[str] | None = None,
            is_prefill: bool = False) -> torch.Tensor

# DecoderLayer.forward 新增 4 个（多一个 layer_idx）：
def forward(self, hidden_states, position_ids, ...,
            paged_kv_cache: PagedKVCache | None = None,
            layer_idx: int | None = None,
            request_ids: list[str] | None = None,
            is_prefill: bool = False) -> torch.Tensor
```

`layer_idx` 不在顶层传入，由 `Qwen3Model` 的 `enumerate(self.layers)` 自动生成：

```python
for i, layer in enumerate(self.layers):
    hidden_states = layer(
        hidden_states, position_ids, ...,
        paged_kv_cache=paged_kv_cache,
        layer_idx=i if paged_kv_cache is not None else None,
        request_ids=request_ids,
        is_prefill=is_prefill,
    )
```

所有新参数都有默认值，M1/M2/M3 路径零影响。

### LLMModel Protocol 扩展

```python
class LLMModel(Protocol):
    def __call__(self, input_ids, *,
                 # ... 现有参数不变 ...
                 paged_kv_cache: object = None,        # M4-T5 新增
                 request_ids: list[str] | None = None,  # M4-T5 新增
                 is_prefill: bool = False,              # M4-T5 新增
                 ) -> torch.Tensor: ...
```

新增参数用默认值，不影响现有 FakeModel 或 M3 调用方。

### Scheduler 方案：inline admission，不新建文件

**不新建** `paged_fcfs.py`。M3 `FCFScheduler` 已稳定，M4 plan §3.1 中 `paged_fcfs.py` 标记为"可能新建"，不是强制。

在 `paged_core.py` 内写一个 `_paged_admit()` helper：

```python
def _paged_admit(scheduler: FCFSScheduler,
                 paged_cache: PagedKVCache) -> list[RequestState]:
    """Block-aware admission：只在 cache 有足够 block 时从 waiting 取请求到 running。

    与 M3 FCFSScheduler.admit_until_full() 的区别：
    - M3 只看 max_num_seqs（slot 数量）
    - M4 检查 can_allocate(prompt_len)，decode 按需分配新 block
    """
    admitted = []
    while scheduler.waiting:
        req = scheduler.waiting[0]
        total_len = req.prompt_ids.shape[1]  # 只为 prompt 预留，decode 按需分配
        if not paged_cache.can_allocate(total_len):
            break  # FCFS：队首不够空间就停，不跳过头部请求
        scheduler.waiting.popleft()
        req.status = RequestStatus.RUNNING
        scheduler.running[req.request_id] = req
        admitted.append(req)
    return admitted
```

T7/T8 再考虑是否抽象公共 admission 接口。

## Block admission 策略

M4 策略：admission 时只按 `prompt_len` 预留，decode 阶段按需分配新 block：

```text
# admission
needed = ceil(prompt_len / block_size)

# decode（每步）
if table.needs_new_block():
    block_id = pool.allocate()  # pool 空时抛 RuntimeError
    table.append_block(block_id)
```

这体现了 paged attention 相对于 M3 fixed-slot 的核心价值：**按需分配，不为未生成的 token 预留空间**。

decode 阶段 block 不足时优雅结束请求（保留已生成 token），不做抢占。高并发场景的动态分配 + preemption 留后续里程碑对齐 vLLM V1 时处理。

本任务不做 chunked prefill，不做复杂 token budget。

## 实现步骤

1. 阅读 M3 `batch_generate` / scheduler / request state，确认 fixed-slot 生命周期。
2. 给 `Qwen3ForCausalLM` / `Qwen3Model` / `DecoderLayer` 加 paged 可选参数（见「接口透传方案」）。
3. 扩展 `LLMModel` Protocol，加 `paged_kv_cache` / `request_ids` / `is_prefill`。
4. 新建 `engine/paged_core.py`，实现 `batch_generate_paged()` 入口。
5. 实现 inline `_paged_admit()`：`can_allocate(prompt_len)` 不满足则 waiting。
6. 实现 paged cache 初始化与 request_id 生成/传递。
7. 实现 batched prefill：多个 admitted 请求 pad + 一次前向 + paged 参数透传。
8. 实现 decode：`append_token` + per-request position_ids + paged decode 参数透传。
9. 实现 finish/free：EOS / max_new_tokens 后 `free_request`。
10. 保留/转发 metrics 基本字段，并新增 paged 相关最小指标（`allocated_blocks` / `free_blocks`）。
11. 补 unit 测试，使用 M3 fixed-slot 作为 oracle（最小闭环验证）。
12. 跑定向测试和 M3 回归。
13. 补齐实现与测试文件教学级注释，任务卡追加完成总结。

## 测试清单

| # | 测什么 | 预期 |
|---|---|---|
| 1 | M3 fixed path 回归 | 原 `batch_generate` 测试继续全绿 |
| 2 | 单请求 paged generate | 输出与 serial/M3 fixed-slot 等价 |
| 3 | 多请求变长 paged generate | paged path 可运行、不崩溃；token 级等价留 T6 全面验证 |
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

- [ ] 模型调用链（`Qwen3ForCausalLM` / `Qwen3Model` / `DecoderLayer`）已加 paged 可选参数，M1/M2/M3 路径不受影响。
- [ ] `LLMModel` Protocol 已扩展 `paged_kv_cache` / `request_ids` / `is_prefill`。
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
4. **模型链透传漏层**：最容易出问题的地方。每加一个参数，必须从 `Qwen3ForCausalLM` → `Qwen3Model` → `DecoderLayer` → `GQAAttention` 全部打通，少一层就 None 传到 attention 层。先用单测锁住透传链路。
5. **把 admission 和 prefix cache 混在一起**：M4 只按容量 admit，不做共享命中。
6. **为了复用 M3 过度改 batch_core**：容易破坏 stable oracle。优先新建 paged 入口。
7. **block 不足时状态半更新**：append/allocate 失败时不能让 request 状态和 cache 状态不一致。
8. **is_prefill 靠 seq_len 猜**：T5 应显式传 `is_prefill`，不靠 token 维度推断。
9. **T5 顺手做 benchmark/attention 重构/文档收口**：这些留给 T6/T7/T8，避免任务失焦。
10. **admission 预留策略的选择**：M4 采用 `prompt_len` 准入 + decode 按需分配，体现 paged 按需分配价值。vLLM V1 的做法是每步动态分配 + 分配失败时抢占低优先级请求（preemption），解决高并发场景的 block 不足问题。动态分配 + preemption 留后续里程碑（M5 prefix cache 或 M10 长上下文调度）对齐 vLLM V1 时引入。

## 与后续任务的衔接

T6 基于 T5 的 paged engine 做 E2E 正确性和 benchmark。T5 完成时只需提供可运行、可测、可解释的 paged batch generation，不要求性能优于 M3。T5 的测试只做最小闭环验证（单请求等价 + 多请求可运行 + admission/free 基本正确），token 级全面等价 + benchmark 留 T6。

## 修订记录

- **2026-08-01**：基于 vLLM V1 架构调研，修订接口透传方案（直接加可选参数，不引入 forward context/bind）、Scheduler 方案（inline admission，不新建 paged_fcfs.py）、参数命名统一为 `is_prefill`、补充 decode position_ids 构造示例、明确 block admission 不预留 decode blocks、调整 T5/T6 测试边界。
