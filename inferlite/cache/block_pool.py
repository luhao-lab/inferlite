"""BlockPool: 物理 KV block 的分配、释放、引用计数和 prefix caching。

M4 版本只管 allocate/free/ref_count。
M5 在此基础上增加：
  - chain hash: 对满 block 的 token 计算链式指纹，支持 prefix cache 查找
  - LRU 淘汰: ref=0 且有 hash 的 block 不立即归还，保留在 LRU 队列等复用
  - touch: cache 命中时把 LRU 中的 block 重新激活（ref++ + 从 LRU 移除）

对齐 vLLM V1: BlockPool + FreeKVCacheBlockQueue
"""

from collections import OrderedDict, deque
from dataclasses import dataclass, field


@dataclass
class Block:
    """物理 block 元数据。

    Attributes:
        block_id: 物理 block 的整数编号，对应 KV tensor 的第 0 维下标。
        hash: 该 block 内容的链式 hash 值（-1 表示未注册）。
            用于 prefix cache 查找：新请求到来时，对 prompt 的满 block
            计算 chain hash，在 hash_to_block_id 中查找是否有相同内容的 block。
        token_ids: 该 block 存储的 token 列表（空列表表示未注册）。
            用于 hash 碰撞二次校验：hash 命中后对比 token_ids 是否真的一致。
        ref_count: 引用计数。
            > 0: 正在被请求使用
            = 0 且 hash != -1: 在 LRU 队列中等待复用
            = 0 且 hash == -1: 在 free pool 中
    """

    block_id: int
    hash: int = -1
    token_ids: list[int] = field(default_factory=list)
    ref_count: int = 0


class BlockPool:
    """管理物理 KV block 的分配、释放、引用计数和 prefix caching。

    BlockPool 只维护 block 元数据，不持有或读写 K/V tensor。

    三个容器管理 block 的生命周期：

        free_block_ids (deque):      真正空闲的 block，可直接分配
        cached_block_lru (OrderedDict): ref=0 但有 hash 的 block，按 LRU 排列
                                        front=最久未用（优先淘汰）
                                        back=最近使用（最后淘汰）
        hash_to_block_id (dict):     chain hash → block_id 的索引

    状态不变量（M5 版本）：
    - ref_count > 0 的 block 不在 free_block_ids 也不在 cached_block_lru
    - ref_count == 0 + hash != -1 → 在 cached_block_lru（等复用）
    - ref_count == 0 + hash == -1 → 在 free_block_ids（真正空闲）
    - allocate() 优先从 free pool 取，不够则淘汰 LRU front
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.num_blocks: int = num_blocks
        self.block_size: int = block_size
        # 所有 block 的元数据
        self.blocks: list[Block] = [Block(block_id=i) for i in range(num_blocks)]
        # 真正空闲的 block 队列（ref=0, 无 hash）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # chain hash → block_id 索引，用于 prefix cache 查找
        self.hash_to_block_id: dict[int, int] = {}
        # LRU 队列：ref=0 有 hash 的 block（保留 KV 数据等复用）
        # OrderedDict 的 key 是 block_id，value 固定为 None
        # move_to_end() 标记最近使用，popitem(last=False) 淘汰最久未用
        self.cached_block_lru: OrderedDict[int, None] = OrderedDict()

    def _validate_block_id(self, block_id: int) -> None:
        """校验物理 block id，避免负数被解释为反向索引。"""
        if not 0 <= block_id < self.num_blocks:
            raise ValueError(f"block_id {block_id} out of range")

    # ── 分配 ──

    def allocate(self) -> int:
        """分配一个 block，ref_count 置为 1。

        分配优先级：
        1. free_block_ids 中的空闲 block（最快，无需清 hash）
        2. cached_block_lru 中最久未用的 block（淘汰 LRU front，清除旧 hash）
        3. 都没有 → RuntimeError
        """
        if self.free_block_ids:
            block_id = self.free_block_ids.popleft()
        elif self.cached_block_lru:
            # 淘汰 LRU 队首（最久未用的 cached block）
            block_id, _ = self.cached_block_lru.popitem(last=False)
            self.reset_hash(block_id)  # 清除旧 hash + hash_to_block_id 映射
        else:
            raise RuntimeError("No free blocks available")

        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.ref_count = 1
        return block_id

    # ── 释放 ──

    def free(self, block_id: int) -> None:
        """释放一个 block（语义上等价于 dec_ref）。"""
        self.dec_ref(block_id)

    def inc_ref(self, block_id: int) -> None:
        """增加一个已分配 block 的引用计数。"""
        self._validate_block_id(block_id)
        block = self.blocks[block_id]
        if block.ref_count == 0:
            raise RuntimeError(f"Block {block_id} is not allocated")
        block.ref_count += 1

    def dec_ref(self, block_id: int) -> None:
        """减少引用计数。ref=0 时按有无 hash 分路：

        有 hash → 进 cached_block_lru 队尾（保留 KV 数据等下次请求复用）
        无 hash → 进 free_block_ids（真正归还空闲池）

        为什么有 hash 不直接归还？因为 hash 意味着这个 block 的内容可能被
        后续请求命中（prefix cache）。保留它在 LRU 中，下次相同前缀的请求
        到来时可以直接 touch 复用，跳过 prefill 计算。
        """
        self._validate_block_id(block_id)
        block = self.blocks[block_id]
        if block.ref_count == 0:
            raise RuntimeError(f"Block {block_id} ref_count already 0")
        block.ref_count -= 1
        if block.ref_count == 0:
            if block.hash != -1:
                # 有 hash → 进 LRU 队尾（最近释放，排到最后）
                self.cached_block_lru[block_id] = None
            else:
                # 无 hash → 直接归还 free pool
                self.free_block_ids.append(block_id)

    @property
    def num_free_blocks(self) -> int:
        """可分配的 block 总数（free pool + LRU 可淘汰）。"""
        return len(self.free_block_ids) + len(self.cached_block_lru)

    # ── Hash 注册与查找 ──

    def hash_blocks(self, block_ids, token_ids, num_full_blocks):
        """prefill/decode 完成后，对填满的 block 计算链式 hash 并注册。

        调用时机：请求完成一轮 prefill 或 decode 后，检查哪些 block 刚好填满，
        计算它们的 chain hash 并写入 hash_to_block_id 索引。

        链式计算：block_i 的 hash = f(block_{i-1}.hash, block_i.token_ids)
        这样相同 token 在不同位置产生不同 hash，避免误命中。

        Args:
            block_ids: 该请求分配的 block id 列表。
            token_ids: 该请求的完整 token 序列。
            num_full_blocks: 需要注册的满 block 数量。
        """
        h = -1  # 第一个 block 无前驱，prefix_hash = -1
        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            h = self.compute_hash(token_ids[start:end], h)
            block = self.blocks[block_ids[i]]
            block.hash = h
            block.token_ids = token_ids[start:end]
            self.hash_to_block_id[h] = block_ids[i]

    @staticmethod
    def compute_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
        """对一个 block 的 token_ids 计算链式 hash。

        用 xxhash64 算法，先喂 prefix_hash 的 8 字节，再喂 token_ids 的字节流。
        链式保证：不同位置的相同 token 产生不同 hash（因为 prefix_hash 不同）。

        Args:
            token_ids: block 内的 token 列表（长度 = block_size）。
            prefix_hash: 前一个 block 的 hash（-1 表示第一个 block）。

        Returns:
            64 位整数 hash 值。
        """
        import numpy as np
        import xxhash

        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    def reset_hash(self, block_id: int) -> None:
        """清除 block 的 hash 注册信息。

        调用时机：allocate() 淘汰 LRU cached block 时，需要清除旧 hash，
        让 hash_to_block_id 不再指向这个 block。
        """
        block = self.blocks[block_id]
        if block.hash != -1:
            del self.hash_to_block_id[block.hash]
            block.hash = -1
            block.token_ids = []

    # ── Prefix Cache 查找与分配 ──

    def can_allocate(self, token_ids_or_count) -> int | bool:
        """检查能否分配 block。兼容 M4 和 M5 两种调用方式。

        M4 路径（传 int）：返回 bool，检查空闲 block 是否够。
        M5 路径（传 list）：查 chain hash 返回命中 block 数（-1 = 容量不够）。

        M5 路径的工作流程：
        1. 把 token_ids 按 block_size 切成满 block
        2. 逐 block 计算 chain hash，在 hash_to_block_id 中查找
        3. 连续命中就累加 hit_count，断了就停
        4. 检查 (总需 block - 命中数) 是否 <= 可用 block
        """
        if isinstance(token_ids_or_count, int):
            # M4 路径：传 block 数量，返回是否够
            num_blocks = token_ids_or_count
            if num_blocks < 0:
                raise ValueError("num_blocks must be non-negative")
            return len(self.free_block_ids) + len(self.cached_block_lru) >= num_blocks

        # M5 路径：传 token_ids，查 prefix cache 命中数
        token_ids = token_ids_or_count
        # 只算满 block（最后一个不满的 block 不参与 hash 查找）
        num_full_blocks = len(token_ids) // self.block_size
        total_needed = (len(token_ids) + self.block_size - 1) // self.block_size

        hit_count = 0
        h = -1
        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            block_tokens = token_ids[start:end]
            h = self.compute_hash(block_tokens, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1:
                break
            # 二次校验：hash 碰撞时 token_ids 可能不同
            if self.blocks[block_id].token_ids != block_tokens:
                break
            hit_count += 1

        # 检查容量：需要 (total_needed - hit_count) 个新 block
        num_new = total_needed - hit_count
        available = len(self.free_block_ids) + len(self.cached_block_lru)
        if available < num_new:
            return -1

        return hit_count

    def touch(self, block_id: int) -> None:
        """prefix cache 命中：重新激活一个 cached block。

        当一个在 LRU 中的 block（ref=0）被新请求命中时：
        1. 从 cached_block_lru 中移除（不再等待淘汰）
        2. ref_count++（标记为有人使用）

        之后该 block 和正常 allocate 的 block 一样使用。
        """
        block = self.blocks[block_id]
        if block.ref_count == 0:
            # 从无引用变有引用 → 从 LRU 移除
            del self.cached_block_lru[block_id]
        block.ref_count += 1

    def allocate_with_cache(self, token_ids: list[int], num_cached: int) -> list[int]:
        """带 prefix cache 的 block 分配。

        前 num_cached 个 block 通过 chain hash 查找复用已有的 cached block，
        剩余 block 从 free pool 新分配。

        工作流程：
        1. 重新计算前 num_cached 个 block 的 chain hash
        2. 从 hash_to_block_id 查到 block_id → touch 复用
        3. 对未命中部分调 allocate() 分配新 block

        Args:
            token_ids: 请求的完整 token 序列。
            num_cached: can_allocate 返回的命中 block 数。

        Returns:
            分配的 block_id 列表（cached + new）。
        """
        block_ids = []
        h = -1

        # 1. 复用 cached block：重算 chain hash → 查索引 → touch
        for i in range(num_cached):
            block_tokens = token_ids[i * self.block_size : (i + 1) * self.block_size]
            h = self.compute_hash(block_tokens, h)
            block_id = self.hash_to_block_id[h]
            self.touch(block_id)
            block_ids.append(block_id)

        # 2. 分配新 block
        total = (len(token_ids) + self.block_size - 1) // self.block_size
        for _ in range(num_cached, total):
            block_ids.append(self.allocate())

        return block_ids
