# M4-T1 — BlockManager

> **状态**：⬜ pending
> **里程碑**：M4 PagedAttention
> **目标**：实现物理 block 的分配、释放、refcount 和 Copy-on-Write 基础能力。

## 背景

M3 的 `SlotManager` 以 request 为单位分配整段连续 KV。M4 改为以 block 为单位分配物理 KV 空间。

## 产出

- `BlockManager`
  - `allocate() -> int`
  - `free(block_id: int) -> None`
  - `inc_ref(block_id: int) -> None`
  - `dec_ref(block_id: int) -> None`
  - `copy_on_write(block_id: int) -> int`
- `Block` 或元数据结构
  - `block_id`
  - `ref_count`

## 算法核心

- `free_block_ids: deque[int]` 保存空闲 block。
- `ref_counts[block_id]` 记录引用计数。
- `allocate()` 从 free list 取 block，refcount 置 1。
- `dec_ref()` 后 refcount 为 0 才归还 free list。
- `copy_on_write()`：如果 refcount == 1，返回原 block；如果 >1，分配新 block，复制旧 block KV，旧 block refcount--。

## 测试

- allocate/free 顺序和数量守恒。
- block 耗尽 raise RuntimeError。
- inc/dec refcount 正确。
- CoW refcount==1 不复制。
- CoW refcount>1 分配新 block，旧 block refcount--，新 block refcount=1。

## DoD

- [ ] `BlockManager` 单测全过。
- [ ] 不依赖 attention/model。
- [ ] 文档说明 block 与 slot 的区别。
