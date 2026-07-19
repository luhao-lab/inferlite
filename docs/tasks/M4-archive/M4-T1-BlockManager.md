# M4-T1 — BlockManager

> **状态**：🔧 in_progress
> **里程碑**：M4 PagedAttention
> **目标**：实现物理 block 的分配、释放、refcount 和 Copy-on-Write 基础能力。

## 背景

M3 的 `SlotManager` 以 request 为单位分配整段连续 KV。M4 改为以 block 为单位分配物理 KV 空间。

M3 的 slot 模型：

```text
request_id -> slot_id
slot_id -> 一整段 [max_seq_len] 连续 KV 空间
```

M4 的 block 模型：

```text
request_id -> block_table -> 多个 physical_block_id
physical_block_id -> 一小段 [block_size] KV 空间
```

T1 只实现 block 级元数据管理，不碰 tensor、不碰 attention、不碰 scheduler。

## 产出

建议新建 `inferlite/model/paged_kv_cache.py`，先只放 T1 需要的元数据类：

```python
@dataclass
class Block:
    block_id: int
    ref_count: int = 0


class BlockManager:
    def __init__(self, num_blocks: int) -> None: ...
    def allocate(self) -> int: ...
    def free(self, block_id: int) -> None: ...
    def inc_ref(self, block_id: int) -> None: ...
    def dec_ref(self, block_id: int) -> None: ...
    def copy_on_write(self, block_id: int) -> int: ...
```

T1 暂不复制 KV tensor。`copy_on_write()` 只负责 refcount 和 block id 语义；真正复制旧 block 的 K/V 放到 T3 `PagedKVCache`，因为 T1 不应该依赖 tensor。

## 接口语义

### `allocate() -> int`

- 从 `free_block_ids` 取一个 block。
- 设置 `ref_count = 1`。
- 返回 `block_id`。
- 如果没有空闲 block，`raise RuntimeError("no free blocks")`。

### `free(block_id: int) -> None`

- 只允许释放 `ref_count == 0` 的 block。
- 将 block 放回 `free_block_ids`。
- 如果 block_id 越界，`raise ValueError`。
- 如果 `ref_count != 0`，`raise RuntimeError`，避免释放仍被引用的 block。

### `inc_ref(block_id: int) -> None`

- `ref_count += 1`。
- 用于后续 Prefix Cache / shared block table。
- block_id 越界要报错。

### `dec_ref(block_id: int) -> None`

- `ref_count -= 1`。
- 如果降到 0，自动进入 free list。
- 如果本来就是 0，`raise RuntimeError`，避免 double free。

### `copy_on_write(block_id: int) -> int`

- 如果 `ref_count == 1`：独占，无需复制，返回原 `block_id`。
- 如果 `ref_count > 1`：
  1. `old.ref_count -= 1`
  2. 分配一个新 block
  3. 新 block `ref_count = 1`
  4. 返回新 `block_id`
- 如果 `ref_count == 0`：`raise RuntimeError`。

T3 会在拿到新 block_id 后复制 K/V tensor；T1 只保证元数据正确。

## 算法核心

- `free_block_ids: deque[int]` 保存空闲 block。
- `blocks: list[Block]` 或 `ref_counts: list[int]` 保存引用计数。
- `allocate()` 从 free list 取 block，refcount 置 1。
- `dec_ref()` 后 refcount 为 0 才归还 free list。
- `copy_on_write()`：如果 refcount == 1，返回原 block；如果 >1，分配新 block，旧 block refcount--。

## 与 nano-vLLM 对齐点

nano-vLLM `BlockManager` 的关键逻辑：

```python
free_block_ids: deque[int]
used_block_ids: set[int]
blocks[block_id].ref_count
```

inferlite M4-T1 对齐：

| nano-vLLM | inferlite M4-T1 |
|---|---|
| `_allocate_block()` | `allocate()` |
| `_deallocate_block()` | `free()` / `dec_ref()` 自动释放 |
| `block.ref_count` | `Block.ref_count` |
| `hash_to_block_id` | 暂不做，留 M5 Prefix Cache |
| `token_ids/hash` | 暂不做，留 M5 |

## 与 vLLM 对齐点

vLLM 的 PagedAttention 依赖两个内存管理事实：

1. physical block 可以被多个 sequence 引用。
2. 写共享 block 前必须 Copy-on-Write。

T1 的 `ref_count` 和 `copy_on_write()` 就是这两个事实的最小可测版本。

## 测试

建议新建 `tests/unit/test_block_manager.py`。

### L0-1 初始化

- `BlockManager(num_blocks=3)`
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

- allocate 得到 block 0
- `dec_ref(0)` 后 refcount=0，block 0 回到 free list
- 再 allocate 应该能拿回 block 0

### L0-5 double free 防御

- refcount 已是 0 时调用 `dec_ref(0)` raise RuntimeError
- refcount 非 0 时直接 `free(0)` raise RuntimeError

### L0-6 inc_ref / dec_ref

- allocate block 0，ref=1
- inc_ref 后 ref=2
- dec_ref 后 ref=1，不释放
- dec_ref 后 ref=0，释放

### L0-7 CoW 独占不复制

- allocate block 0，ref=1
- `copy_on_write(0)` 返回 0
- free list 不变，ref=1

### L0-8 CoW 共享时复制

- allocate block 0，ref=1
- inc_ref block 0，ref=2
- `copy_on_write(0)` 返回新 block 1
- block 0 ref=1
- block 1 ref=1

### L0-9 invalid block id

- `inc_ref(-1)` / `inc_ref(num_blocks)` raise ValueError
- `dec_ref(-1)` / `free(num_blocks)` raise ValueError

## DoD

- [ ] `BlockManager` 单测全过。
- [ ] 不依赖 attention/model/tensor。
- [ ] 不修改 M3 `BatchedKVCache`。
- [ ] 文档说明 block 与 slot 的区别。
- [ ] `docs/tasks/M4-archive/M4-T1-BlockManager.md` 末尾追加完成总结。
