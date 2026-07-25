# M4-T1 — BlockPool

> 实现 PagedAttention 的最小物理 block 元数据池：负责分配、释放和引用计数，不持有 KV tensor。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T1 |
| 里程碑 | M4 — PagedAttention |
| 状态 | 🔧 in-progress |
| 前置 | M3 — Continuous Batching ✅ |
| 后续 | M4-T2 — BlockTable |
| 估时 | 1.5h |
| 核心文件 | `inferlite/cache/block_pool.py` |

## 目标

### 要解决什么问题

M3 的 `SlotManager` 以 request 为单位分配整段连续 KV：

```text
request_id -> slot_id
slot_id -> 一整段 [max_seq_len] 连续 KV 空间
```

M4 改为以 block 为单位管理物理 KV 空间：

```text
request_id -> block_table -> 多个 physical_block_id
physical_block_id -> 一小段 [block_size] KV 空间
```

T1 先建立最小 block 元数据池，为后续 BlockTable 和 PagedKVCache 提供稳定的分配接口。

### 做完是什么效果

```python
pool = BlockPool(num_blocks=3, block_size=16)
block_id = pool.allocate()
assert block_id == 0
assert pool.blocks[block_id].ref_count == 1

pool.free(block_id)
assert pool.num_free_blocks == 3
```

### 不做什么

- 不持有或读写 K/V tensor。
- 不实现 BlockTable、PagedKVCache、attention 和 scheduler。
- 不实现 Copy-on-Write；partial-hit CoW 留到 M5。
- 不实现 prefix hash、LRU 和 eviction；这些属于 M5 Prefix Caching。
- 不耦合 `RequestState` / Sequence，请求到 block 的映射由 T2 管理。

### 在推理链路中的位置

```text
M3 RequestState / Scheduler
          |
          v
M4-T3 PagedKVCache
     |             |
     v             v
M4-T2 BlockTable   M4-T1 BlockPool
逻辑位置映射       物理 block 分配
```

### 设计原则

- **M4 不含 CoW**：beam search 不在引擎内 fork；M5 在上层以独立请求实现。
- **保留 ref_count**：建立引用计数不变量，为 M5 的 block 共享打基础。
- **BlockPool 只管元数据**：不耦合请求状态和 tensor。

## 产出文件

- `inferlite/cache/block_pool.py::Block`
- `inferlite/cache/block_pool.py::BlockPool`
- `tests/unit/test_block_pool.py`

## 接口骨架

```python
@dataclass
class Block:
    block_id: int
    ref_count: int = 0


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int) -> None: ...
    def _validate_block_id(self, block_id: int) -> None: ...
    def allocate(self) -> int: ...
    def free(self, block_id: int) -> None: ...
    def inc_ref(self, block_id: int) -> None: ...
    def dec_ref(self, block_id: int) -> None: ...
    def can_allocate(self, num_blocks: int) -> bool: ...

    @property
    def num_free_blocks(self) -> int: ...
```

M4 的 `Block` 只保存 `block_id/ref_count`，不提前放置 prefix cache 字段。M5 再新增 `block_hash/token_ids` 等缓存元数据。

T1 不复制 KV tensor，也不实现 `copy_on_write()`。M5 的 partial-hit CoW 需要拷贝 K/V，职责属于 `PagedKVCache`，不属于只管理元数据的 BlockPool。

## 接口合同

### `__init__(num_blocks: int, block_size: int)`

- `num_blocks`：物理 block 总数，必须是正整数。
- `block_size`：每个 block 容纳的 token 数，必须是正整数。
- 任一参数小于等于 0 时，`raise ValueError`。
- 初始化 `blocks: list[Block]` 和 `free_block_ids: deque[int]`。

### `_validate_block_id(block_id: int) -> None`

- 统一校验 `0 <= block_id < num_blocks`。
- 非法 id（包括 Python 会解释成反向索引的负数）抛 `ValueError`。
- `free()` / `inc_ref()` / `dec_ref()` 复用该校验，避免异常合同分散。

### `allocate() -> int`

- 从 `free_block_ids` 队头取一个 block。
- 取出的 block 必须满足 `ref_count == 0`；否则说明内部状态损坏，抛 `RuntimeError`。
- 设置 `ref_count = 1`。
- 返回 `block_id`。
- 如果没有空闲 block，`raise RuntimeError("No free blocks available")`。

### `free(block_id: int) -> None`

- 请求释放一个 block 引用；实现上等价于 `dec_ref(block_id)`。
- ref_count 减 1，降为 0 时归还空闲池。
- block_id 越界，`raise ValueError`。
- ref_count 已经是 0，显式 `raise RuntimeError`，避免 double free。
- 请求生命周期代码优先调用 `free()`；`dec_ref()` 是底层引用计数原语。

### `inc_ref(block_id: int) -> None`

- 仅允许增加**已分配 block**（`ref_count > 0`）的引用计数。
- `ref_count += 1`。
- 用于 M5 Prefix Cache 命中已在用 block 等共享场景。
- block_id 越界时 `raise ValueError`。
- block 仍在空闲池（`ref_count == 0`）时 `raise RuntimeError`，避免同一 block 同时处于空闲和被引用状态。
- M5 若要重新激活 ref_count=0 的缓存 block，应通过独立 `touch()` 将其先从淘汰队列移除，不复用 M4 的 `inc_ref()`。

### `dec_ref(block_id: int) -> None`

- `ref_count -= 1`。
- 如果降到 0，自动进入 free list。
- block_id 越界时 `raise ValueError`。
- 如果本来就是 0，显式 `raise RuntimeError`，避免 double free。

### `can_allocate(num_blocks: int) -> bool`

- 检查是否有足够空闲 block。
- `num_blocks < 0` 时 `raise ValueError`。
- `num_blocks == 0` 时返回 `True`。
- 其他情况返回 `len(free_block_ids) >= num_blocks`。

### `num_free_blocks -> int`（property）

- 返回当前空闲 block 数。

## 算法核心与不变量

### 数据结构

- `free_block_ids: deque[int]`：保存空闲 block，`popleft()` / `append()` 均为 O(1)。
- `blocks: list[Block]`：按 block_id 直接索引元数据。

### 状态不变量

```text
block_id in free_block_ids  <=>  blocks[block_id].ref_count == 0
blocks[block_id].ref_count > 0  =>  block_id not in free_block_ids
```

- `allocate()` 从 free list 取 block，将 ref_count 从 0 置为 1。
- `dec_ref()` 只在 ref_count 从 1 降到 0 时归还 free list。
- `inc_ref()` 只能增加已分配 block 的引用，不能重新激活空闲 block。

## 实现步骤

1. 给 `BlockPool.__init__` 加入 `block_size`，并校验两个构造参数。
2. 实现 `_validate_block_id()`，统一负数和越界检查。
3. 收敛 `allocate/free/inc_ref/dec_ref` 的状态转换与显式异常。
4. 实现 `can_allocate()` 和 `num_free_blocks`。
5. 删除旧的 `copy_on_write()` 以及 fork/beam search 相关注释。
6. 运行 T1 单测，再运行全量回归测试。

## 复杂度

| 操作 | 时间复杂度 | 原因 |
|---|---:|---|
| `allocate()` | O(1) | `deque.popleft()` |
| `free()` / `dec_ref()` | O(1) | list 索引 + `deque.append()` |
| `inc_ref()` | O(1) | list 索引 |
| `can_allocate()` / `num_free_blocks` | O(1) | `len(deque)` |
| 初始化 | O(num_blocks) | 创建 Block 列表和 free deque |

## 与 vLLM 系实现的关系

| 实现 | 对应能力 | inferlite M4-T1 |
|---|---|---|
| vLLM V1 `BlockPool.get_new_blocks()` | 从 free queue 分配 block，ref_cnt 0→1 | `allocate()`（单 block 教学版） |
| vLLM V1 `BlockPool.free_blocks()` | ref_cnt--，归还 free queue | `free()` / `dec_ref()` |
| vLLM V1 `BlockPool.touch()` | prefix hit 后增加引用 | M5 再实现；M4 仅保留受限 `inc_ref()` 原语 |
| vLLM V1 `num_gpu_blocks` | GPU 物理 block 总数 | `num_blocks`（教学版不区分 GPU/CPU） |
| nano-vLLM `BlockManager` | allocation + prefix hash | M4 只取 allocation/ref_count，hash 留 M5 |
| Prefix hash / eviction / CoW | 缓存复用与共享写入 | M5 |

## L0 测试清单

| # | 测什么 | 预期 |
|---|---|---|
| 1 | 初始化 | free ids 为 `[0, 1, 2]`，所有 ref=0 |
| 2 | allocate 顺序与状态 | 依次返回 0、1、2；分配后 ref=1，且不在 free list |
| 3 | pool 耗尽 | 再次 allocate 抛 `RuntimeError` |
| 4 | dec_ref 自动释放 | 单 block pool 释放后可立即复用 |
| 5 | double free | `free/dec_ref` 抛 `RuntimeError` |
| 6 | 引用计数 | `1 -> 2 -> 1 -> 0`；空闲 block 不允许 inc_ref |
| 7 | can_allocate | 容量变化正确；0 为 True；负数抛 `ValueError` |
| 8 | 非法 block id | 负数和越界值抛 `ValueError` |
| 9 | 构造参数 | `block_size` 保存正确；非正参数抛 `ValueError` |

### 测试命令

```bash
uv run pytest tests/unit/test_block_pool.py -q
```

<details>
<summary>展开查看测试场景细节</summary>

### L0-1 初始化

- `BlockPool(num_blocks=3, block_size=16)`
- `free_block_ids == [0, 1, 2]`
- 所有 `ref_count == 0`

### L0-2 allocate 顺序

- 连续 allocate 三次返回 `0, 1, 2`
- 三个 block refcount 都是 1
- free list 为空

### L0-3 耗尽报错

- `num_blocks=1`
- 第二次 `allocate()` raise RuntimeError

### L0-4 dec_ref 自动释放

- `num_blocks=1`，allocate 得到 block 0。
- `dec_ref(0)` 后 refcount=0，block 0 回到 free list。
- 再 allocate 能拿回 block 0。

### L0-5 double free 防御

- refcount 已是 0 时调用 `dec_ref(0)` raise RuntimeError。
- `free(0)` 在 refcount=0 时也 raise RuntimeError。

### L0-6 inc_ref / dec_ref

- allocate block 0，ref=1。
- inc_ref 后 ref=2。
- dec_ref 后 ref=1，不释放。
- dec_ref 后 ref=0，释放。
- 对从未 allocate 的空闲 block 调用 inc_ref raise RuntimeError。

### L0-7 can_allocate

- num_blocks=3。
- can_allocate(3) == True。
- allocate 三次后 can_allocate(1) == False。
- dec_ref 一个后 can_allocate(1) == True。
- can_allocate(0) == True。
- can_allocate(-1) raise ValueError。

### L0-8 invalid block id

- `inc_ref(-1)` / `inc_ref(num_blocks)` raise ValueError
- `dec_ref(-1)` / `free(num_blocks)` raise ValueError

### L0-9 构造参数与 block_size

- `BlockPool(num_blocks=10, block_size=16).block_size == 16`。
- `BlockPool(num_blocks=10, block_size=32).block_size == 32`。
- `num_blocks <= 0` / `block_size <= 0` raise ValueError。

</details>

## DoD

- [ ] `BlockPool` 实现与任务卡接口合同一致。
- [ ] `uv run pytest tests/unit/test_block_pool.py -q` 全绿。
- [ ] `uv run pytest tests/ -q` 全量回归通过。
- [ ] free-list 与 ref_count 状态一致：空闲 block 必须 ref=0，ref>0 的 block 不在 free list。
- [ ] 所有公开方法使用显式异常，不依赖 `assert` 做运行时校验。
- [ ] 不依赖 attention/model/tensor。
- [ ] 不修改 M3 `BatchedKVCache`。
- [ ] 不含 prefix cache 元数据和 `copy_on_write`（移 M5）。
- [ ] 末尾追加 `## 完成总结`。
- [ ] commit：`feat(kv-cache): add physical block pool (M4-T1 done)`。

## 坑（按概率排序）

1. **负数 block_id 被 Python 当成反向索引**：所有公开 block-id 接口都应先做显式范围校验。
2. **对空闲 block 调用 inc_ref**：会破坏 free-list/ref_count 一致性，必须拒绝。
3. **用 assert 做运行时校验**：`python -O` 会移除 assert，改用显式异常。
4. **误以为释放后一定立即复用**：deque 使用 `append`，多 block 时释放项进入队尾。
5. **提前实现 M5**：T1 不添加 hash、token_ids、LRU 或 CoW。
