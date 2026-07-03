# M3-T2 BatchedKVCache + SlotManager

> M3 第二张任务卡：把 M2 的单请求 KV Cache 推进到固定槽位的多请求 KV Cache。

## 元信息
- **任务 ID**: M3-T2
- **里程碑**: M3 — Continuous Batching
- **状态**: ⬜ pending
- **前置**: M3-T1
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
slot = cache.allocate("req-1")
assert slot == 0
cache.seq_lens[slot] = 128
cache.free(slot)
assert cache.is_free(slot)
```

**不做什么（边界）**：

- 不实现 PagedAttention。
- 不做 block table。
- 不做 prefix cache / session cache / TTL。
- 不做 LRU / eviction。
- 不改 attention kernel。
- 不做跨请求共享 KV。

## 产出文件

- `inferlite/model/batched_kv_cache.py::BatchedLayerKVCache`
- `inferlite/model/batched_kv_cache.py::BatchedKVCache`
- `inferlite/model/batched_kv_cache.py::SlotManager`
- `tests/unit/test_batched_kv_cache.py`

## 算法核心

```python
@dataclass
class BatchedLayerKVCache:
    # [S, n_kv_heads, L, head_dim]
    k: torch.Tensor
    v: torch.Tensor


class SlotManager:
    def __init__(self, max_num_slots: int) -> None:
        self.free_slots: list[int] = list(range(max_num_slots))
        self.req_to_slot: dict[str, int] = {}
        self.slot_to_req: dict[int, str] = {}

    def allocate(self, request_id: str) -> int:
        ...

    def free(self, slot_id: int) -> None:
        ...


class BatchedKVCache:
    def __init__(self, layers: list[BatchedLayerKVCache], max_seq_len: int) -> None:
        self.layers = layers
        self.max_seq_len = max_seq_len
        self.max_num_slots = layers[0].k.shape[0]
        self.seq_lens = torch.zeros(self.max_num_slots, dtype=torch.long)
        self.occupied = torch.zeros(self.max_num_slots, dtype=torch.bool)
        self.slot_manager = SlotManager(self.max_num_slots)
```

M3 固定槽位 shape：

```python
k_cache.shape = [max_num_slots, n_kv_heads, max_seq_len, head_dim]
v_cache.shape = [max_num_slots, n_kv_heads, max_seq_len, head_dim]
```

`max_num_slots` 和 `cache_slots` 的区别：

```text
max_num_slots: 预分配槽位总数，也是最大 running 请求数
cache_slots: 当前 decode step 参与 batch 的 slot id 列表，shape [B]
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|---|---|---|
| 1 | cache shape | `[S, H_kv, L, D]` | 精确 |
| 2 | dtype/device 继承 config | 与输入一致 | 精确 |
| 3 | allocate 顺序 | 默认从低 slot id 开始 | 精确 |
| 4 | 分配超过容量 | 抛 `RuntimeError` | 精确 |
| 5 | free 后可复用 | freed slot 能再次 allocate | 精确 |
| 6 | duplicate request_id | 抛 `ValueError` | 精确 |
| 7 | free 非 occupied slot | 抛 `ValueError` | 精确 |
| 8 | seq_lens 初始化/重置 | slot free 后 seq_len=0 | 精确 |
| 9 | occupied mask | allocate/free 后一致 | 精确 |

## DoD

- [ ] `BatchedKVCache.from_config()` 可按 M2 config 创建固定槽位 cache。
- [ ] `SlotManager` 支持 allocate/free/query。
- [ ] `seq_lens` / `occupied` 状态和 slot 生命周期一致。
- [ ] 单测覆盖容量、复用、异常、不变量。
- [ ] `uv run pytest tests/unit/test_batched_kv_cache.py -q` 通过。
- [ ] 不修改 M2 `KVCache` 行为，避免破坏已有测试。
- [ ] commit `feat(kv-cache): add fixed-slot batched KV cache (M3-T2 done)`。

## 坑（按概率排序）

1. **误把 slot 数等同总请求数**：waiting queue 可以大于 `max_num_slots`，slot 只限制 running 请求。
2. **释放 slot 时忘记清 `seq_lens`**：下一请求会继承旧长度。
3. **为了简单直接覆盖 M2 KVCache**：会破坏 M2 generate；M3 应新增 batched cache。
4. **提前做 PagedAttention**：M3 目标是 fixed-slot 教学版，分页属于 M4。

## 完成总结

待完成后补：固定槽位 KV 池的接口、shape、生命周期和后续 T3 attention 依赖。
