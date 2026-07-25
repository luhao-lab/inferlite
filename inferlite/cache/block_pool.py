from collections import deque
from dataclasses import dataclass


@dataclass
class Block:
    """物理 block 元数据。"""

    block_id: int
    ref_count: int = 0


class BlockPool:
    """管理物理 KV block 的分配、释放和引用计数。

    BlockPool 只维护 block 元数据，不持有或读写 K/V tensor。

    状态不变量：
    - ref_count == 0 的 block 位于 free_block_ids。
    - ref_count > 0 的 block 不位于 free_block_ids。
    - allocate() 将 ref_count 从 0 变为 1。
    - dec_ref() 将 ref_count 减到 0 时归还 block。
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.num_blocks: int = num_blocks
        self.block_size: int = block_size
        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    def _validate_block_id(self, block_id: int) -> None:
        """校验物理 block id，避免负数被解释为反向索引。"""
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(f"block_id {block_id} out of range")

    def allocate(self) -> int:
        """分配一个空闲 block，ref_count 置为 1。无空闲时抛 RuntimeError。"""
        if not self.free_block_ids:
            raise RuntimeError("No free blocks available")

        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.ref_count = 1
        return block_id

    def free(self, block_id: int) -> None:
        """释放一个 block（语义上等价于 dec_ref）。

        ref_count 减 1；若降为 0 则归还空闲池。
        """
        self.dec_ref(block_id)

    def inc_ref(self, block_id: int) -> None:
        """增加一个已分配 block 的引用计数。"""
        self._validate_block_id(block_id)
        block = self.blocks[block_id]
        if block.ref_count == 0:
            raise RuntimeError(f"Block {block_id} is not allocated")
        block.ref_count += 1

    def dec_ref(self, block_id: int) -> None:
        """减少引用计数。降为 0 时自动归还空闲池。"""
        self._validate_block_id(block_id)
        block = self.blocks[block_id]
        if block.ref_count == 0:
            raise RuntimeError(f"Block {block_id} ref_count already 0")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_ids.append(block_id)

    def can_allocate(self, num_blocks: int) -> bool:
        """检查是否还能分配 block。"""
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        return len(self.free_block_ids) >= num_blocks

    @property
    def num_free_blocks(self) -> int:
        """返回当前可分配的物理 block 数。"""
        return len(self.free_block_ids)
