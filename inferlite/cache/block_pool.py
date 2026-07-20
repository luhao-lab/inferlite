from collections import deque
from dataclasses import dataclass


@dataclass
class Block:
    """物理 block 元数据。"""

    block_id: int
    ref_count: int = 0


class BlockPool:
    """物理 block 的分配与引用计数管理。

    只管理元数据（哪个 block 空闲、被引用几次），不持有 K/V tensor。
    K/V 数据的实际拷贝由调用方（PagedKVCache）在 copy_on_write 返回后完成。
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    def allocate(self) -> int:
        """分配一个空闲 block，ref_count 置为 1。无空闲时抛 RuntimeError。"""
        if not self.free_block_ids:
            raise RuntimeError("No free blocks available")
        block_id = self.free_block_ids.popleft()
        self.blocks[block_id].ref_count = 1
        return block_id

    def free(self, block_id: int) -> None:
        """释放一个 block（语义上等价于 dec_ref）。

        ref_count 减 1；若降为 0 则归还空闲池。
        """
        self.dec_ref(block_id)

    def inc_ref(self, block_id: int) -> None:
        """增加引用计数（fork / beam search 时调用）。"""
        self.blocks[block_id].ref_count += 1

    def dec_ref(self, block_id: int) -> None:
        """减少引用计数。降为 0 时自动归还空闲池。"""
        block = self.blocks[block_id]
        assert block.ref_count > 0, f"Block {block_id} ref_count already 0"
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_block_ids.append(block_id)

    def copy_on_write(self, block_id: int) -> int:
        """写时复制（元数据部分）。

        - ref_count == 1 → 无需复制，直接返回原 block_id。
        - ref_count > 1  → 分配新 block、旧 block dec_ref，返回新 block_id。
          调用方负责将旧 block 的 K/V tensor 拷贝到新 block。
        """
        if self.blocks[block_id].ref_count == 1:
            return block_id
        new_block_id = self.allocate()
        self.dec_ref(block_id)
        return new_block_id
