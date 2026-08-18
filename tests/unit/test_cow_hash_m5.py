"""M5-T3 Partial Hit CoW + hash 注册测试。

T3 目标：
  1. hash_blocks：prefill/decode 后注册填满 block 的 chain hash
  2. cow_if_shared：shared block 写入前先拷贝独占副本 + hash 迁移
"""

import pytest
import torch

from inferlite.cache.paged_kv_cache import PagedKVCache
from inferlite.config import ModelConfig

# ── Fixtures ──


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        vocab_size=32,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


@pytest.fixture
def cache(config: ModelConfig) -> PagedKVCache:
    return PagedKVCache.from_config(
        config, num_blocks=8, block_size=4, dtype=torch.float32, device="cpu"
    )


def _make_prompt(n: int, base: int = 1) -> list[int]:
    return list(range(base, base + n))


def _write_kv(cache: PagedKVCache, request_id: str, value: float = 1.0) -> None:
    """给请求的 block 写入 KV 数据。"""
    table = cache.block_tables[request_id]
    for layer in cache.layers:
        for bid in table.block_ids:
            layer.k[bid] = torch.full_like(layer.k[bid], value)
            layer.v[bid] = torch.full_like(layer.v[bid], value + 100)


# ── 1. hash_blocks 注册 ──


class TestHashBlocksRegistration:
    """hash_blocks 在 PagedKVCache 层面的注册逻辑。"""

    def test_register_full_blocks(self, cache: PagedKVCache):
        """prefill 后注册满 block 的 chain hash。"""
        tokens = _make_prompt(8)  # 2 full blocks
        cache.allocate_request("a", 8)
        table = cache.block_tables["a"]

        cache.hash_blocks("a", tokens)

        # 两个 block 都应有 hash
        b0 = cache.block_pool.blocks[table.block_ids[0]]
        b1 = cache.block_pool.blocks[table.block_ids[1]]
        assert b0.hash != -1
        assert b1.hash != -1
        assert b0.hash != b1.hash  # chain hash 不同
        # hash_to_block_id 应能查到
        assert cache.block_pool.hash_to_block_id[b0.hash] == table.block_ids[0]
        assert cache.block_pool.hash_to_block_id[b1.hash] == table.block_ids[1]

    def test_skip_already_hashed_blocks(self, cache: PagedKVCache):
        """已注册的 block 不重复注册（hash 不变）。"""
        tokens = _make_prompt(4)  # 1 full block
        cache.allocate_request("a", 4)

        # 第一次注册
        cache.hash_blocks("a", tokens)
        bid = cache.block_tables["a"].block_ids[0]
        first_hash = cache.block_pool.blocks[bid].hash

        # 第二次注册（应该跳过）
        cache.hash_blocks("a", tokens)
        second_hash = cache.block_pool.blocks[bid].hash

        assert first_hash == second_hash

    def test_partial_block_not_hashed(self, cache: PagedKVCache):
        """不满的 block 不注册 hash。"""
        tokens = _make_prompt(5)  # 1 full + 1 partial
        cache.allocate_request("a", 5)
        table = cache.block_tables["a"]

        cache.hash_blocks("a", tokens)

        # 第一个 block 有 hash，第二个没有
        b0 = cache.block_pool.blocks[table.block_ids[0]]
        b1 = cache.block_pool.blocks[table.block_ids[1]]
        assert b0.hash != -1
        assert b1.hash == -1


# ── 2. CoW (Copy-on-Write) ──


class TestCowIfShared:
    """shared block 写入前的 CoW 拷贝。"""

    def test_cow_not_needed_for_exclusive_block(self, cache: PagedKVCache):
        """独占 block（ref=1）不需要 CoW，返回原 block_id。"""
        tokens = _make_prompt(4)
        cache.allocate_request("a", 4)
        bid = cache.block_tables["a"].block_ids[0]

        new_bid = cache.cow_if_shared("a", block_idx=0)
        assert new_bid == bid  # 不变

    def test_cow_creates_exclusive_copy(self, cache: PagedKVCache):
        """shared block（ref>1）CoW 后产生独占副本。"""
        tokens = _make_prompt(4)

        # 1. 请求 A 分配 + 写 KV + 注册 hash
        cache.allocate_request("a", 4)
        _write_kv(cache, "a", value=10.0)
        cache.hash_blocks("a", tokens)
        old_bid = cache.block_tables["a"].block_ids[0]

        # 2. 释放 A（block 进 LRU，ref=0，但 hash 还在）
        cache.free_request("a")

        # 3. 请求 B touch 复用（ref=1）
        cache.allocate_request_with_cache("b", tokens, num_cached=1)
        assert cache.block_tables["b"].block_ids[0] == old_bid

        # 4. 请求 C 也 touch 同一个 block（模拟 ref=2 的 shared 场景）
        cache.block_pool.touch(old_bid)  # ref=2
        # 手动把 C 的 block_table 也指向这个 block
        from inferlite.cache.paged_kv_cache import BlockTable

        table_c = BlockTable(request_id="c", block_size=4)
        table_c.append_block(old_bid)
        table_c.extend(4)
        cache.block_tables["c"] = table_c

        # 5. 对 B 执行 CoW
        new_bid = cache.cow_if_shared("b", block_idx=0)
        assert new_bid != old_bid  # 新 block
        assert cache.block_tables["b"].block_ids[0] == new_bid

        # 6. 新 block 独占（ref=1）
        assert cache.block_pool.blocks[new_bid].ref_count == 1
        # 旧 block ref 减了 1（B 不再持有）
        assert cache.block_pool.blocks[old_bid].ref_count == 1  # C 仍持有

    def test_cow_preserves_kv_data(self, cache: PagedKVCache):
        """CoW 后新 block 的 KV 数据与原 block 一致。"""
        tokens = _make_prompt(4)

        # 分配 + 写 KV + hash
        cache.allocate_request("a", 4)
        _write_kv(cache, "a", value=42.0)
        cache.hash_blocks("a", tokens)
        old_bid = cache.block_tables["a"].block_ids[0]

        # 保存 KV 数据
        saved_k = cache.layers[0].k[old_bid].clone()
        saved_v = cache.layers[0].v[old_bid].clone()

        # 制造 shared 场景：touch 让 ref=2
        cache.block_pool.touch(old_bid)
        # CoW
        new_bid = cache.cow_if_shared("a", block_idx=0)

        # 新 block KV == 旧 block KV
        assert torch.equal(cache.layers[0].k[new_bid], saved_k)
        assert torch.equal(cache.layers[0].v[new_bid], saved_v)

    def test_cow_migrates_hash(self, cache: PagedKVCache):
        """CoW 后 hash_to_block_id 指向新 block。"""
        tokens = _make_prompt(4)

        cache.allocate_request("a", 4)
        _write_kv(cache, "a", value=7.0)
        cache.hash_blocks("a", tokens)
        old_bid = cache.block_tables["a"].block_ids[0]
        old_hash = cache.block_pool.blocks[old_bid].hash

        # 制造 shared + CoW
        cache.block_pool.touch(old_bid)
        new_bid = cache.cow_if_shared("a", block_idx=0)

        # hash_to_block_id 应指向新 block
        assert cache.block_pool.hash_to_block_id[old_hash] == new_bid
        # 新 block 有 hash，旧 block 的 hash 被清掉
        assert cache.block_pool.blocks[new_bid].hash == old_hash
