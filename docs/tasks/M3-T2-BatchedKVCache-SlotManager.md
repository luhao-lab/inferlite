# M3-T2 BatchedKVCache + SlotManager

> M3 第二张任务卡：把 M2 的单请求 KV Cache 推进到固定槽位的多请求 KV Cache。

## 元信息
- **任务 ID**: M3-T2
- **里程碑**: M3 — Continuous Batching
- **状态**: 🔧 in_progress
- **前置**: M3-T1 ✅
- **估时**: 3h

## 目标

**要解决什么问题**：

M2 的 `KVCache` 只有一个全局 `cur_len`，适合单请求：

```text
single request → one cache → one cur_len
```

M3 要支持多个请求同时 decode，需要固定大小的 KV slot 池：

```text
max_num_slots = S
slot 0 → req_a, seq_len=128
slot 1 → req_b, seq_len=64
slot 2 → free
...
```

每个 running request 占用一个 slot，每个 slot 独立维护实际长度。

**做完是什么效果**：

可以分配、释放、复用 slot，并验证 KV Cache shape：

```python
cache = BatchedKVCache.from_config(..., max_num_slots=8, max_seq_len=1024)
slot = cache.slot_manager.allocate("req-1")
assert slot == 0
cache.seq_lens[slot] = 128
cache.slot_manager.free("req-1")
assert cache.slot_manager.is_free(slot)
```

**不做什么（边界）**：

- 不实现 PagedAttention（M4）
- 不做 block table / prefix cache / session cache / TTL
- 不做 LRU / eviction
- 不改 attention kernel
- 不做跨请求共享 KV
- **不修改 M2 的 `KVCache`**（新增文件，M2 保持兼容）

## 产出文件

- `inferlite/model/batched_kv_cache.py::BatchedLayerKVCache`
- `inferlite/model/batched_kv_cache.py::SlotManager`
- `inferlite/model/batched_kv_cache.py::BatchedKVCache`
- `tests/unit/test_batched_kv_cache.py`

## 算法核心

### BatchedLayerKVCache

和 M2 的 `LayerKVCache` 结构一样，只是第一维从 `batch_size` 变为 `max_num_slots`：

```python
@dataclass
class BatchedLayerKVCache:
    k: torch.Tensor  # [S, n_kv_heads, max_seq_len, head_dim]
    v: torch.Tensor  # 同上
```

### SlotManager

管理 slot 的分配和释放。只保留 `req_to_slot` 单向映射（调用方有 request_id，不需要反向查找）：

```python
class SlotManager:
    def __init__(self, max_num_slots: int) -> None:
        self.max_num_slots = max_num_slots
        self.free_slots: deque[int] = deque(range(max_num_slots))  # deque, popleft O(1)
        self.req_to_slot: dict[str, int] = {}                      # 唯一映射

    def allocate(self, request_id: str) -> int:
        """分配一个 slot。
        - request_id 重复 → raise ValueError
        - free_slots 空 → raise RuntimeError（防御性检查）
        """
        ...

    def free(self, request_id: str) -> None:
        """释放 slot。传入 request_id（调用方有 RequestState 对象）。
        - request_id 不在 req_to_slot → raise ValueError
        """
        ...

    def is_free(self, slot_id: int) -> bool:
        """slot_id 是否空闲。"""
        ...
```

### BatchedKVCache

组装多层 cache + per-slot 状态 + SlotManager：

```python
class BatchedKVCache:
    def __init__(
        self,
        layers: list[BatchedLayerKVCache],
        max_seq_len: int,
        max_num_slots: int,
    ) -> None:
        self.layers = layers
        self.max_seq_len = max_seq_len
        self.max_num_slots = max_num_slots
        # seq_lens: 每个 slot 的当前有效长度（= prompt_len + num_generated）
        self.seq_lens = torch.zeros(max_num_slots, dtype=torch.long)
        # occupied: 每个 slot 是否被占用（和 SlotManager.req_to_slot 对应）
        self.occupied = torch.zeros(max_num_slots, dtype=torch.bool)
        self.slot_manager = SlotManager(max_num_slots)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        max_num_slots: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "BatchedKVCache":
        """按模型配置创建固定槽位 cache。参考 M2 的 KVCache.from_config()。"""
        ...
```

### 关键设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| free_slots 数据结构 | deque | popleft O(1) |
| 映射方向 | 只保留 req_to_slot | 调用方有 request_id，free 传 request_id 即可 |
| free 时是否清零 tensor | 不清零 | 和 M2 reset() 一致，下次 prefill 覆盖写入 |
| free 时清什么 | seq_lens[slot]=0, occupied[slot]=False | 保持状态一致 |
| slot 耗尽 | raise RuntimeError | 防御性检查，正常不应发生 |
| request_id 重复 | raise ValueError | 调用方逻辑错误 |

### M3 shape 对比

```python
# M2
k_cache.shape = [batch_size=1, n_kv_heads, max_seq_len, head_dim]
# M3
k_cache.shape = [max_num_slots, n_kv_heads, max_seq_len, head_dim]
```

`max_num_slots` 和 `cache_slots` 的区别：

```text
max_num_slots: 预分配槽位总数，也是最大 running 请求数
cache_slots: 当前 decode step 参与 batch 的 slot id 列表（T3 attention 用）
```

## 实现步骤

1. 写 `BatchedLayerKVCache`（最简单，和 M2 LayerKVCache 一样）
2. 写 `SlotManager`（allocate / free / is_free，用 deque）
3. 写 `BatchedKVCache`（from_config + seq_lens + occupied + SlotManager）
4. 写单测 `test_batched_kv_cache.py`

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|---|---|---|
| 1 | cache shape | `[S, H_kv, L, D]` | 精确 |
| 2 | dtype/device 继承 config | 与输入一致 | 精确 |
| 3 | allocate 顺序 | 默认从低 slot id 开始 | 精确 |
| 4 | 分配超过容量 | 抛 `RuntimeError` | 精确 |
| 5 | free 后可复用 | freed slot 能再次 allocate | 精确 |
| 6 | duplicate request_id | 抛 `ValueError` | 精确 |
| 7 | free 不存在的 request_id | 抛 `ValueError` | 精确 |
| 8 | seq_lens 初始化/重置 | slot free 后 seq_len=0 | 精确 |
| 9 | occupied mask | allocate/free 后一致 | 精确 |

## DoD

- [ ] `BatchedKVCache.from_config()` 可按 config 创建固定槽位 cache。
- [ ] `SlotManager` 支持 allocate / free / is_free。
- [ ] `seq_lens` / `occupied` 状态和 slot 生命周期一致。
- [ ] 单测覆盖 L0 清单全部 9 项。
- [ ] `uv run pytest tests/unit/test_batched_kv_cache.py -q` 通过。
- [ ] 不修改 M2 `KVCache` 行为，已有测试不受影响。
- [ ] commit `feat(kv-cache): add fixed-slot batched KV cache (M3-T2 done)`。

## 坑（按概率排序）

1. **误把 slot 数等同总请求数**：waiting queue 可以大于 `max_num_slots`，slot 只限制 running 请求。
2. **释放 slot 时忘记清 `seq_lens`**：下一请求会继承旧长度。
3. **free 时清零了 tensor**：不需要，和 M2 reset() 一样只清元数据。
4. **覆盖 M2 KVCache**：会破坏 M2 generate；M3 应新增文件。
5. **提前做 PagedAttention**：M3 目标是 fixed-slot 教学版，分页属于 M4。

## 完成总结

待完成后补：固定槽位 KV 池的接口、shape、生命周期和后续 T3 attention 依赖。
