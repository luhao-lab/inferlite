# M4-T1 — BlockPool

> **状态**：🔧 in_progress
> **里程碑**：M4 PagedAttention
> **目标**：实现物理 block 的分配、释放、ref_count 基础能力。不含 CoW（移 M5）。

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

### 设计原则（基于 vLLM V1）

- **M4 不含 CoW**：beam search 不在引擎内 fork，在上层以独立请求方式实现（学 vLLM V1）
- **ref_count 保留**：为 M5 prefix caching 预留基础设施
- **BlockPool 保持纯元数据**：不耦合 Sequence/RequestState

## 产出

文件：`inferlite/cache/block_pool.py`

```python
@dataclass
class Block:
    block_id: int
    ref_count: int = 0


class BlockPool:
    def __init__(self, num_blocks: int, block_size: int) -> None: ...
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

## 接口语义

### `__init__(num_blocks: int, block_size: int)`

<!-- codeflicker-fix: EDGE-Issue-004/stva3u29f3klfevrjaqn -->

- `num_blocks`：物理 block 总数，必须是正整数。
- `block_size`：每个 block 容纳的 token 数，必须是正整数。
- 任一参数小于等于 0 时，`raise ValueError`。
- 初始化 `blocks: list[Block]` 和 `free_block_ids: deque[int]`。

### `allocate() -> int`

- 从 `free_block_ids` 取一个 block。
- 设置 `ref_count = 1`。
- 返回 `block_id`。
- 如果没有空闲 block，`raise RuntimeError("No free blocks available")`。

### `free(block_id: int) -> None`

<!-- codeflicker-fix: API-Issue-002/stva3u29f3klfevrjaqn -->

- 请求释放一个 block 引用；实现上等价于 `dec_ref(block_id)`。
- ref_count 减 1，降为 0 时归还空闲池。
- block_id 越界，`raise ValueError`。
- ref_count 已经是 0，显式 `raise RuntimeError`，避免 double free。
- 请求生命周期代码优先调用 `free()`；`dec_ref()` 是底层引用计数原语。

### `inc_ref(block_id: int) -> None`

<!-- codeflicker-fix: LOGIC-Issue-001/stva3u29f3klfevrjaqn -->

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

<!-- codeflicker-fix: EDGE-Issue-005/stva3u29f3klfevrjaqn -->

- 检查是否有足够空闲 block。
- `num_blocks < 0` 时 `raise ValueError`。
- `num_blocks == 0` 时返回 `True`。
- 其他情况返回 `len(free_block_ids) >= num_blocks`。

### `num_free_blocks -> int`（property）

- 返回当前空闲 block 数。

## 算法核心

- `free_block_ids: deque[int]` 保存空闲 block。
- `blocks: list[Block]` 保存元数据（block_id + ref_count）。
- `allocate()` 从 free list 取 block，refcount 置 1。
- `dec_ref()` 后 refcount 为 0 才归还 free list。

## 与 nano-vLLM / vLLM V1 对齐点

| nano-vLLM / vLLM V1 | inferlite M4-T1 |
|---|---|
| `BlockPool._allocate_block()` | `allocate()` |
| `BlockPool._deallocate_block()` | `free()` / `dec_ref()` 自动释放 |
| `block.ref_count` | `Block.ref_count` |
| `num_gpu_blocks` | `num_blocks`（教学版不区分 GPU/CPU） |
| `hash_to_block_id` | 暂不做，留 M5 Prefix Cache |
| `copy_on_write` | 暂不做，留 M5 partial hit CoW |

## 测试

建议新建 `tests/unit/test_block_pool.py`。

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

<!-- codeflicker-fix: LOGIC-Issue-003/stva3u29f3klfevrjaqn -->

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

## DoD

- [ ] `BlockPool` 单测全过。
- [ ] free-list 与 ref_count 状态一致：空闲 block 必须 ref=0，ref>0 的 block 不在 free list。
- [ ] 所有公开方法使用显式异常，不依赖 `assert` 做运行时校验。
- [ ] 不依赖 attention/model/tensor。
- [ ] 不修改 M3 `BatchedKVCache`。
- [ ] 不含 prefix cache 元数据和 `copy_on_write`（移 M5）。
- [ ] `docs/tasks/M4-archive/M4-T1-BlockPool.md` 末尾追加完成总结。
