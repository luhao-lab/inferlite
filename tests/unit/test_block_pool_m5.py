"""M5-T1: BlockPool hash + LRU 单测。

测试 BlockPool 升级后的 chain hash、LRU 淘汰、touch、can_allocate、
allocate_with_cache、hash_blocks 等能力。
"""

import pytest

from inferlite.cache.block_pool import Block, BlockPool

# ── compute_hash ──


class TestComputeHash:
    """chain hash 确定性与位置唯一性。"""

    def test_deterministic(self):
        """相同 token 序列产生相同 hash。"""
        tokens = [1, 2, 3, 4]
        h1 = BlockPool.compute_hash(tokens)
        h2 = BlockPool.compute_hash(tokens)
        assert h1 == h2

    def test_different_tokens_different_hash(self):
        """不同 token 序列产生不同 hash。"""
        h1 = BlockPool.compute_hash([1, 2, 3, 4])
        h2 = BlockPool.compute_hash([5, 6, 7, 8])
        assert h1 != h2

    def test_chain_hash_position_unique(self):
        """相同 token 在不同位置产生不同 hash（链式保证）。"""
        tokens = [1, 2, 3, 4]
        # block 0: prefix_hash = -1
        h0 = BlockPool.compute_hash(tokens, prefix_hash=-1)
        # block 1: prefix_hash = h0
        h1 = BlockPool.compute_hash(tokens, prefix_hash=h0)
        assert h0 != h1

    def test_chain_hash_prefix_matters(self):
        """prefix_hash 不同 → hash 不同。"""
        tokens = [1, 2, 3, 4]
        h_a = BlockPool.compute_hash(tokens, prefix_hash=100)
        h_b = BlockPool.compute_hash(tokens, prefix_hash=200)
        assert h_a != h_b


# ── Block 扩展 ──


class TestBlock:
    """Block dataclass 新增字段。"""

    def test_block_has_hash_and_token_ids(self):
        b = Block(block_id=0)
        assert b.hash == -1
        assert b.token_ids == []


# ── touch ──


class TestTouch:
    """cache 命中时 touch() 行为。"""

    def test_touch_increments_ref(self):
        pool = BlockPool(num_blocks=4, block_size=4)
        # 分配 block 0
        bid = pool.allocate()
        assert pool.blocks[bid].ref_count == 1
        # touch → ref++
        pool.touch(bid)
        assert pool.blocks[bid].ref_count == 2

    def test_touch_cached_block_removes_from_lru(self):
        """touch 一个在 LRU 中的 block → 从 LRU 移除 + ref=1。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        # 分配并释放（带 hash → 进 LRU）
        bid = pool.allocate()
        pool.blocks[bid].hash = 12345
        pool.blocks[bid].token_ids = [1, 2, 3, 4]
        pool.hash_to_block_id[12345] = bid
        pool.dec_ref(bid)
        # 应该在 LRU 中
        assert bid in pool.cached_block_lru
        # touch → 从 LRU 移除 + ref=1
        pool.touch(bid)
        assert bid not in pool.cached_block_lru
        assert pool.blocks[bid].ref_count == 1


# ── LRU 淘汰 ──


class TestLRU:
    """free 时 LRU 行为。"""

    def test_free_with_hash_goes_to_lru(self):
        """ref=0 + 有 hash → 进 cached_block_lru，不进 free_pool。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        bid = pool.allocate()
        pool.blocks[bid].hash = 999
        pool.hash_to_block_id[999] = bid
        pool.dec_ref(bid)
        assert bid in pool.cached_block_lru
        assert bid not in pool.free_block_ids

    def test_free_without_hash_goes_to_free_pool(self):
        """ref=0 + 无 hash → 进 free_pool，不进 LRU。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        bid = pool.allocate()
        # 不设置 hash
        pool.dec_ref(bid)
        assert bid in pool.free_block_ids
        assert bid not in pool.cached_block_lru

    def test_allocate_evicts_lru_when_no_free(self):
        """free pool 空 → 淘汰 LRU 最久未用的 block。"""
        pool = BlockPool(num_blocks=2, block_size=4)
        # 分配全部 block
        b0 = pool.allocate()
        b1 = pool.allocate()
        assert pool.num_free_blocks == 0
        # b0 设 hash 并释放 → 进 LRU
        pool.blocks[b0].hash = 111
        pool.blocks[b0].token_ids = [1, 2, 3, 4]
        pool.hash_to_block_id[111] = b0
        pool.dec_ref(b0)
        # b1 无 hash 释放 → 进 free pool
        pool.dec_ref(b1)
        # 现在 free_pool=[b1], LRU=[b0]
        # allocate 优先用 free pool
        got1 = pool.allocate()
        assert got1 == b1
        # 再 allocate → 淘汰 LRU 的 b0
        got2 = pool.allocate()
        assert got2 == b0
        # b0 的 hash 应被清除
        assert pool.blocks[b0].hash == -1
        assert 111 not in pool.hash_to_block_id

    def test_lru_eviction_order(self):
        """最久未用的先淘汰。"""
        pool = BlockPool(num_blocks=3, block_size=4)
        # 分配全部
        ids = [pool.allocate() for _ in range(3)]
        # 全部设 hash 并释放（按 0, 1, 2 顺序）
        for i, bid in enumerate(ids):
            pool.blocks[bid].hash = i + 100
            pool.hash_to_block_id[i + 100] = bid
            pool.blocks[bid].token_ids = [i]
            pool.dec_ref(bid)
        # LRU 顺序：0(最久), 1, 2(最近)
        # touch(0) → 0 移到队尾 → 顺序变为 1, 2, 0
        pool.touch(ids[0])
        # 释放 touch 增加的 ref
        pool.dec_ref(ids[0])
        # 现在 LRU 顺序：1(最久), 2, 0
        # 分配新 block → 淘汰 1
        got = pool.allocate()
        assert got == ids[1]

    def test_allocate_no_free_raises(self):
        """无 free 且无 LRU → RuntimeError。"""
        pool = BlockPool(num_blocks=2, block_size=4)
        pool.allocate()
        pool.allocate()
        with pytest.raises(RuntimeError):
            pool.allocate()


# ── can_allocate ──


class TestCanAllocate:
    """查 chain hash 返回命中 block 数。"""

    def test_no_cache_returns_zero(self):
        """无缓存时返回 0。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        result = pool.can_allocate([1, 2, 3, 4, 5, 6, 7, 8])
        assert result == 0

    def test_cache_hit(self):
        """有缓存时返回命中 block 数。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        # 模拟已注册的 hash
        tokens = [1, 2, 3, 4]
        h = BlockPool.compute_hash(tokens)
        bid = pool.allocate()
        pool.blocks[bid].hash = h
        pool.blocks[bid].token_ids = tokens
        pool.hash_to_block_id[h] = bid
        # 查询相同 token → 命中 1 block
        result = pool.can_allocate(tokens + [5, 6, 7, 8])
        assert result == 1

    def test_cache_miss_different_tokens(self):
        """token 不同 → 不命中。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        tokens = [1, 2, 3, 4]
        h = BlockPool.compute_hash(tokens)
        bid = pool.allocate()
        pool.blocks[bid].hash = h
        pool.blocks[bid].token_ids = tokens
        pool.hash_to_block_id[h] = bid
        # 不同 token
        result = pool.can_allocate([9, 9, 9, 9])
        assert result == 0

    def test_insufficient_space_returns_negative(self):
        """空闲不够 → 返回 -1。"""
        pool = BlockPool(num_blocks=2, block_size=4)
        # 占满
        pool.allocate()
        pool.allocate()
        result = pool.can_allocate([1, 2, 3, 4, 5, 6, 7, 8])
        assert result == -1


# ── allocate_with_cache ──


class TestAllocateWithCache:
    """touch cached block + 分配新 block。"""

    def test_allocate_with_cached_blocks(self):
        """分配时复用 cached block。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        # 注册 block 0 的 hash
        tokens_a = [1, 2, 3, 4]
        h = BlockPool.compute_hash(tokens_a)
        b0 = pool.allocate()
        pool.blocks[b0].hash = h
        pool.blocks[b0].token_ids = tokens_a
        pool.hash_to_block_id[h] = b0
        pool.dec_ref(b0)  # ref=0, 进 LRU
        # allocate_with_cache: 1 cached + 1 new
        tokens_full = [1, 2, 3, 4, 5, 6, 7, 8]
        block_ids = pool.allocate_with_cache(tokens_full, num_cached=1)
        assert len(block_ids) == 2
        assert block_ids[0] == b0  # cached
        assert pool.blocks[b0].ref_count == 1  # touch 后 ref=1

    def test_allocate_with_no_cache(self):
        """num_cached=0 → 全部分配新 block。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        block_ids = pool.allocate_with_cache(tokens, num_cached=0)
        assert len(block_ids) == 2
        assert all(pool.blocks[bid].ref_count == 1 for bid in block_ids)


# ── hash_blocks ──


class TestHashBlocks:
    """prefill/decode 后注册满 block hash。"""

    def test_hash_blocks_registers(self):
        """注册后 hash_to_block_id 有对应条目。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        b0 = pool.allocate()
        b1 = pool.allocate()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        pool.hash_blocks([b0, b1], tokens, num_full_blocks=2)
        assert pool.blocks[b0].hash != -1
        assert pool.blocks[b1].hash != -1
        assert pool.blocks[b0].hash in pool.hash_to_block_id
        assert pool.blocks[b1].hash in pool.hash_to_block_id

    def test_hash_blocks_chain(self):
        """链式 hash：block 1 的 hash 依赖 block 0。"""
        pool = BlockPool(num_blocks=4, block_size=4)
        b0 = pool.allocate()
        b1 = pool.allocate()
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]
        pool.hash_blocks([b0, b1], tokens, num_full_blocks=2)
        # 手动验证 chain
        h0 = BlockPool.compute_hash([1, 2, 3, 4], prefix_hash=-1)
        h1 = BlockPool.compute_hash([5, 6, 7, 8], prefix_hash=h0)
        assert pool.blocks[b0].hash == h0
        assert pool.blocks[b1].hash == h1


# ── M4 回归 ──


class TestM4Regression:
    """无 hash 时行为与 M4 完全一致。"""

    def test_basic_allocate_free(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        b0 = pool.allocate()
        b1 = pool.allocate()
        assert pool.num_free_blocks == 2
        pool.free(b0)
        assert pool.num_free_blocks == 3
        pool.free(b1)
        assert pool.num_free_blocks == 4

    def test_ref_count(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        b0 = pool.allocate()
        pool.inc_ref(b0)
        assert pool.blocks[b0].ref_count == 2
        pool.dec_ref(b0)
        assert pool.blocks[b0].ref_count == 1
        pool.dec_ref(b0)
        assert pool.blocks[b0].ref_count == 0

    def test_can_allocate_simple(self):
        pool = BlockPool(num_blocks=4, block_size=16)
        assert pool.can_allocate(3) is True
        pool.allocate()
        assert pool.can_allocate(3) is True
        pool.allocate()
        assert pool.can_allocate(3) is False
