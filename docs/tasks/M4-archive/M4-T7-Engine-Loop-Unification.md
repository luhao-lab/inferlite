# M4-T7 — Engine Loop Unification + CacheAdapter Protocol

> 消除 `batch_core.py`（M3 fixed-slot）和 `paged_core.py`（M4 paged）之间 80% 的重复代码：抽取公共 batch generation 主循环到 `engine/loop.py`，用 `CacheAdapter` Protocol 消化 cache 差异，让两个入口文件变成 ~30 行的薄包装。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T7 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ⬜ pending |
| 前置 | M4-T6 E2E Correctness & Benchmark ✅ |
| 后续 | M4-T8 Attention Backend Refactor + 模型链瘦身 |
| 估时 | 3～4h |
| 核心文件 | `inferlite/engine/loop.py`（新建）、`inferlite/cache/adapter.py`（新建） |
| 改造文件 | `inferlite/engine/batch_core.py`、`inferlite/engine/paged_core.py` |
| 测试文件 | 既有 M3/M4 batch 测试 |
| 产物 | 公共主循环 + CacheAdapter Protocol + 两个薄包装入口 |

## 背景

M4-T5 为了最小闭环，把 `paged_core.py` 写成 `batch_core.py` 的翻版。两者主循环结构几乎相同：

```text
while scheduler.has_unfinished():
    admit → prefill → decode → sample → finish/free
```

只有 cache 操作不同：

| 环节 | batch_core (M3) | paged_core (M4) |
|---|---|---|
| admission | `admit_until_full()` | `_paged_admit()` + `can_allocate` |
| allocate | `allocate_slot()` | `allocate_request()` |
| position | `cache.seq_lens[slots]` | `block_tables[rid].seq_len` |
| append | `seq_lens[slot] += 1` | `append_token()` |
| model params | `kv_cache` + `cache_slots` + `cache_positions` | `paged_kv_cache` + `request_ids` + `is_prefill` |
| free | `free_slot()` | `free_request()` |

如果 M5 Prefix Cache、M10 Chunked Prefill 继续复制粘贴新循环，engine 层会越来越乱。T7 的职责是在 M4 收口前，把公共逻辑抽出来。

## 范围冻结

T7 是 **结构整理任务**，不是新能力任务。

### 明确做

- 抽取公共 `batch_generate_loop()` 到 `engine/loop.py`。
- 新增 `CacheAdapter` Protocol 到 `cache/adapter.py`。
- 实现 `BatchedCacheAdapter`（包装 M3 BatchedKVCache）。
- 实现 `PagedCacheAdapter`（包装 M4 PagedKVCache）。
- `batch_core.py` 瘦身为薄包装：创建 cache + adapter + scheduler，调 `batch_generate_loop()`。
- `paged_core.py` 瘦身为薄包装：同上。
- **paged 路径统一传 `position_embeddings`**：`Qwen3Model.forward` paged 分支改为传 `position_embeddings=position_embeddings`（与 non-paged 分支一致），消除每层 attention 重算 RoPE 的冗余和 bug 风险。
- 保持 M3/M4 旧路径行为不变。
- 跑既有 M3/M4 batch 测试回归。

### 明确不做

- 不重构 `GQAAttention` 或模型链（T8）。
- 不修改 `LLMModel` Protocol（T8 可能调整）。
- 不实现 Prefix Cache / LRU / CoW（M5）。
- 不做 kernel backend（M9）。
- 不删除 M3 fixed-slot 路径（它仍是 oracle）。
- 不修改 scheduler 内部逻辑（admission 差异由 adapter 消化）。

## 核心设计

### 1. CacheAdapter Protocol

```python
# cache/adapter.py

from typing import Protocol

class CacheAdapter(Protocol):
    """Engine loop 与 cache 实现之间的适配层。

    engine/loop.py 只通过这个接口操作 cache，
    不关心底层是 BatchedKVCache 还是 PagedKVCache。
    """

    def can_admit(self, request: RequestState) -> bool:
        """检查 cache 是否有空间接受该请求。"""
        ...

    def allocate(self, request: RequestState) -> None:
        """为请求分配 cache 空间（prefill 前调用）。"""
        ...

    def prefill_model_kwargs(self, request_ids: list[str]) -> dict:
        """返回 prefill model forward 需要的额外 keyword arguments。

        M3: {"kv_cache": cache, "cache_slots": slots}
        M4: {"paged_kv_cache": cache, "request_ids": rids, "is_prefill": True}
        """
        ...

    def prepare_decode(self, request_id: str) -> None:
        """decode forward 前准备：分配 cache 空间（如需要），但不推进 seq_len。

        M3: 无操作（fixed-slot 已在 prefill 时分配）。
        M4: 检查是否需要新 block，需要则分配。
        """
        ...

    def decode_positions(self, request_ids: list[str]) -> list[int]:
        """返回每个请求当前的 decode position（0-indexed 绝对位置）。

        必须在 prepare_decode 之后、model forward 之前调用。
        返回值直接作为 position_ids，不需要再 -1。
        """
        ...

    def commit_decode(self, request_ids: list[str]) -> None:
        """model forward 成功后推进 seq_len。

        M3: seq_lens[slot] += 1
        M4: block_table.extend(1)
        """
        ...

    def decode_model_kwargs(self, request_ids: list[str]) -> dict:
        """返回 decode model forward 需要的额外 keyword arguments。"""
        ...

    def free(self, request_id: str) -> None:
        """释放请求的 cache 空间。"""
        ...
```

### 2. BatchedCacheAdapter

```python
class BatchedCacheAdapter:
    """M3 fixed-slot BatchedKVCache 的 adapter。"""

    def __init__(self, cache: BatchedKVCache, max_num_seqs: int) -> None:
        self.cache = cache
        self.max_num_seqs = max_num_seqs

    def can_admit(self, request: RequestState) -> bool:
        # M3 admission: max_num_seqs 控制
        return len(self.cache.occupied_slots) < self.max_num_seqs

    def allocate(self, request: RequestState) -> None:
        slot = self.cache.allocate_slot(request.request_id)
        request.slot_id = slot

    def prefill_model_kwargs(self, request_ids: list[str]) -> dict:
        slots = [self.cache.slot_map[rid] for rid in request_ids]
        return {"kv_cache": self.cache, "cache_slots": torch.tensor(slots)}

    def prepare_decode(self, request_id: str) -> None:
        pass  # fixed-slot 已在 prefill 分配，无需额外操作

    def decode_positions(self, request_ids: list[str]) -> list[int]:
        return [int(self.cache.seq_lens[self.cache.slot_map[rid]]) for rid in request_ids]

    def commit_decode(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            slot = self.cache.slot_map[rid]
            self.cache.seq_lens[slot] += 1

    def decode_model_kwargs(self, request_ids: list[str]) -> dict:
        slots = [self.cache.slot_map[rid] for rid in request_ids]
        positions = [int(self.cache.seq_lens[s]) for s in slots]
        return {
            "kv_cache": self.cache,
            "cache_slots": torch.tensor(slots),
            "cache_positions": torch.tensor(positions),
        }

    def free(self, request_id: str) -> None:
        self.cache.free_slot(request_id)
```

### 3. PagedCacheAdapter

```python
class PagedCacheAdapter:
    """M4 paged PagedKVCache 的 adapter。"""

    def __init__(self, cache: PagedKVCache) -> None:
        self.cache = cache

    def can_admit(self, request: RequestState) -> bool:
        # M4 admission: can_allocate(prompt_len)
        prompt_len = request.prompt_ids.shape[1]
        return self.cache.can_allocate(prompt_len)

    def allocate(self, request: RequestState) -> None:
        prompt_len = request.prompt_ids.shape[1]
        self.cache.allocate_request(request.request_id, prompt_len)

    def prefill_model_kwargs(self, request_ids: list[str]) -> dict:
        return {
            "paged_kv_cache": self.cache,
            "request_ids": request_ids,
            "is_prefill": True,
        }

    def prepare_decode(self, request_id: str) -> None:
        table = self.cache.block_tables[request_id]
        if table.needs_new_block():
            block_id = self.cache.block_pool.allocate()
            table.append_block(physical_block_id=block_id)

    def decode_positions(self, request_ids: list[str]) -> list[int]:
        # seq_len 在 prepare_decode 阶段未 +1，直接就是正确的 0-indexed position
        return [self.cache.block_tables[rid].seq_len for rid in request_ids]

    def commit_decode(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            self.cache.block_tables[rid].extend(1)

    def decode_model_kwargs(self, request_ids: list[str]) -> dict:
        return {
            "paged_kv_cache": self.cache,
            "request_ids": request_ids,
            "is_prefill": False,
        }

    def free(self, request_id: str) -> None:
        self.cache.free_request(request_id)
```

### 4. 公共主循环

```python
# engine/loop.py

def batch_generate_loop(
    model: LLMModel,
    sampler: GreedySampler,
    scheduler: FCFSScheduler,
    cache_adapter: CacheAdapter,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
    metrics: MetricsCollector | None = None,
) -> list[torch.Tensor]:
    """M3/M4 共享的 batch generation 主循环。

    cache 差异全部由 cache_adapter 消化，loop 本身不区分 fixed-slot 和 paged。

    主循环结构：
    while scheduler.has_unfinished():
        admit → allocate → batched prefill → sample
        decode: prepare_decode → decode_positions → model forward → commit_decode → sample → finish/free
    """
    # ... 约 100 行，从当前 paged_core.py 抽取 ...
```

### 5. 入口文件变薄

```python
# batch_core.py（~30行）
def batch_generate(model, sampler, prompts, max_new_tokens,
                   max_num_slots, config, max_seq_len, ...):
    cache = BatchedKVCache.from_config(config, max_num_slots, max_seq_len, ...)
    adapter = BatchedCacheAdapter(cache, max_num_slots)
    scheduler = FCFSScheduler(max_num_seqs=max_num_slots)
    # submit prompts to scheduler...
    return batch_generate_loop(model, sampler, scheduler, adapter, prompts, max_new_tokens, ...)

# paged_core.py（~30行）
def batch_generate_paged(model, sampler, prompts, max_new_tokens,
                         num_blocks, block_size, config, ...):
    cache = PagedKVCache.from_config(config, num_blocks, block_size, ...)
    adapter = PagedCacheAdapter(cache)
    scheduler = FCFSScheduler(max_num_seqs=num_blocks)
    # submit prompts to scheduler...
    return batch_generate_loop(model, sampler, scheduler, adapter, prompts, max_new_tokens, ...)
```

## 与 vLLM 的关系

| vLLM 概念 | inferlite T7 对应 |
|---|---|
| `GPUModelRunner.execute_model()` | `engine/loop.py batch_generate_loop()` |
| `KVCacheManager` 接口 | `CacheAdapter` Protocol |
| `V1KVCacheManager` / `V0KVCacheManager` | `PagedCacheAdapter` / `BatchedCacheAdapter` |
| scheduler + model runner 分离 | scheduler 不变，loop 统一调度+前向 |

## 实现步骤

1. 记录当前 M3/M4 batch 测试命令，作为重构前基线。
2. 新建 `cache/adapter.py`：定义 `CacheAdapter` Protocol + `BatchedCacheAdapter` + `PagedCacheAdapter`。
3. 新建 `engine/loop.py`：从 `paged_core.py` 抽取公共主循环，cache 操作替换为 adapter 调用。
4. 改造 `batch_core.py`：瘦身为薄包装，调用 `batch_generate_loop` + `BatchedCacheAdapter`。
5. 改造 `paged_core.py`：瘦身为薄包装，调用 `batch_generate_loop` + `PagedCacheAdapter`。
6. 跑 M3 batch 测试回归（`test_batch_engine.py`）。
7. 跑 M4 paged 测试回归（`test_paged_batch_engine.py`）。
8. 跑 M3 fixed-slot 全量回归。
9. 更新必要注释：说明 cache adapter 的设计思路。
10. 任务卡追加完成总结与 commit 号。

## 测试要求

至少运行：

```bash
uv run pytest tests/unit/test_batch_engine.py -q
uv run pytest tests/unit/test_paged_batch_engine.py -q
uv run pytest tests/unit/test_paged_attention.py -q
uv run pytest tests/unit/test_paged_kv_cache.py -q
```

所有既有测试必须全绿。

## DoD

- [ ] `engine/loop.py` 存在，包含公共 `batch_generate_loop()`。
- [ ] `cache/adapter.py` 存在，包含 `CacheAdapter` Protocol + 两个 adapter 实现。
- [ ] `batch_core.py` 瘦身为薄包装（< 50 行）。
- [ ] `paged_core.py` 瘦身为薄包装（< 50 行）。
- [ ] M3 fixed-slot 路径行为不变，测试全绿。
- [ ] M4 paged 路径行为不变，测试全绿。
- [ ] M5 Prefix Cache 可以只实现一个新的 `PrefixCacheAdapter` 就接入 engine loop。
- [ ] 任务卡完成总结记录真实命令、结果和 commit。

## 坑（按概率排序）

1. **prefill/decode 的 model kwargs 接口不统一**：M3 传 `kv_cache` + `cache_slots`，M4 传 `paged_kv_cache` + `request_ids`。adapter 的 `prefill_model_kwargs` / `decode_model_kwargs` 返回 dict，loop 里用 `**kwargs` 展开传给 model。要确保 LLMModel Protocol 接受这些参数。
2. **prefill batch 构建逻辑差异**：M3 逐条 prefill，M4 batched prefill（pad 到最长）。adapter 可能需要暴露 `supports_batched_prefill` 标志，或让 loop 统一走 batched prefill。
3. **scheduler admission 差异**：M3 的 `admit_until_full` 在 scheduler 内部做，M4 的 `_paged_admit` 在 engine 层做。统一到 loop 里需要让 admission 调用 adapter.can_admit()。
4. **decode 时序差异（重要）**：M3 是“先读 seq_lens，再 forward，最后 +1”；M4 原来是“先 append_token(+1)，再 forward”。T7 统一为主流时序：“prepare_decode（分配空间）→ decode_positions（读位置）→ forward → commit_decode(+1)”。M4 的 PagedCacheAdapter 把原来的 append_token 拆成 prepare_decode + commit_decode 两步，彻底消除 off-by-1 风险。
4. **metrics 字段差异**：M3 有 `occupied_slots`，M4 可以报告 `allocated_blocks`。adapter 可以暴露 `metrics_snapshot()` 方法。
5. **一次性改太多**：先保证 M3 回归通过，再验证 M4。不要同时改两个入口。
6. **T7 顺手做 attention 重构或模型链瘦身**：这些留给 T8，避免任务失焦。

## 与 T8 / M5 的衔接

- **T8**（Attention Backend Refactor + 模型链瘦身）：T7 的 adapter 返回 dict，T8 可以改成返回 `AttentionCacheContext`，进一步收敛 model 参数。
- **M5**（Prefix Cache）：只需新增一个 `PrefixCacheAdapter`（包装带 hash/LRU 的 PagedKVCache），engine loop 不变。
- **M10**（Chunked Prefill）：adapter 的 `can_admit` 可以检查 chunked 分配，loop 不变。
