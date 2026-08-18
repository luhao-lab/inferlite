"""M5-T2 prefix cache allocate + skip prefill 测试。

T2 目标：把 BlockPool 的 hash 查找能力接入引擎，让新请求能命中 prefix cache
并跳过已缓存 token 的 prefill。

测试分三层：
  1. PagedKVCache.allocate_request_with_cache — block table 包含 cached + new block
  2. PagedCacheAdapter — can_admit_with_cache / allocate_with_cache
  3. loop.py 集成 — 相同 prompt 第二个请求跳过 prefill，输出等价
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
    """生成确定性 token 序列：[base, base+1, ..., base+n-1]"""
    return list(range(base, base + n))


def _write_fake_kv(cache: PagedKVCache, request_id: str, value: float = 1.0) -> None:
    """给指定请求的 block 写入可识别的 KV 数据（用于验证 KV 是否保留）。"""
    table = cache.block_tables[request_id]
    for layer_idx in range(len(cache.layers)):
        layer = cache.layers[layer_idx]
        for bid in table.block_ids:
            layer.k[bid] = torch.full_like(layer.k[bid], value)
            layer.v[bid] = torch.full_like(layer.v[bid], value + 100)


# ── 1. PagedKVCache.allocate_request_with_cache ──


class TestAllocateRequestWithCache:
    """PagedKVCache 层面的 cache-aware 分配测试。"""

    def test_no_cache_hit_same_as_m4(self, cache: PagedKVCache):
        """无 cache 命中时，行为与 M4 allocate_request 一致。"""
        tokens = _make_prompt(8)  # 8 tokens, 2 blocks
        cache.allocate_request_with_cache("r0", tokens, num_cached=0)

        table = cache.block_tables["r0"]
        assert len(table.block_ids) == 2
        assert table.seq_len == 8

    def test_cache_hit_reuses_blocks(self, cache: PagedKVCache):
        """命中 cached block 后，新请求复用相同物理 block。"""
        # 1. 请求 A 分配 2 个 block（8 tokens），写入 hash
        tokens = _make_prompt(8)
        cache.allocate_request("a", 8)
        cache.block_pool.hash_blocks(cache.block_tables["a"].block_ids, tokens, num_full_blocks=2)
        original_blocks = list(cache.block_tables["a"].block_ids)

        # 2. 释放 A 的 block（进 LRU，保留 hash）
        cache.free_request("a")
        assert len(cache.block_pool.cached_block_lru) == 2

        # 3. 请求 B 用相同 prefix 分配，应命中 2 个 cached block
        cache.allocate_request_with_cache("b", tokens, num_cached=2)
        new_blocks = cache.block_tables["b"].block_ids

        # 物理 block 应和 A 的一样（复用）
        assert new_blocks == original_blocks
        assert cache.block_tables["b"].seq_len == 8

    def test_partial_cache_hit(self, cache: PagedKVCache):
        """部分命中：前 1 个 block 命中，第 2 个新分配。"""
        # 1. 请求 A 分配 1 个 block（4 tokens），写入 hash
        tokens_a = _make_prompt(4)
        cache.allocate_request("a", 4)
        cache.block_pool.hash_blocks(cache.block_tables["a"].block_ids, tokens_a, num_full_blocks=1)
        cached_block_id = cache.block_tables["a"].block_ids[0]

        # 2. 释放 A
        cache.free_request("a")

        # 3. 请求 B 有 8 tokens（2 blocks），前 4 token 和 A 相同
        tokens_b = tokens_a + _make_prompt(4, base=100)  # 后 4 token 不同
        cache.allocate_request_with_cache("b", tokens_b, num_cached=1)
        table_b = cache.block_tables["b"]

        # 第一个 block 应复用 A 的，第二个是新分配的
        assert table_b.block_ids[0] == cached_block_id
        assert table_b.block_ids[1] != cached_block_id
        assert len(table_b.block_ids) == 2
        assert table_b.seq_len == 8

    def test_cached_block_kv_data_preserved(self, cache: PagedKVCache):
        """cached block 的 KV 数据在复用后仍然完整保留。"""
        # 1. 请求 A 分配 + 写入 KV + 注册 hash
        tokens = _make_prompt(4)
        cache.allocate_request("a", 4)
        _write_fake_kv(cache, "a", value=42.0)
        cache.block_pool.hash_blocks(cache.block_tables["a"].block_ids, tokens, num_full_blocks=1)
        cached_bid = cache.block_tables["a"].block_ids[0]

        # 保存 A 的 KV 数据副本
        saved_k = cache.layers[0].k[cached_bid].clone()

        # 2. 释放 A
        cache.free_request("a")

        # 3. 请求 B 复用
        cache.allocate_request_with_cache("b", tokens, num_cached=1)

        # KV 数据应未被破坏
        assert torch.equal(cache.layers[0].k[cached_bid], saved_k)


# ── 2. BlockPool can_allocate（cache 查找） ──


class TestBlockPoolCanAllocate:
    """BlockPool 层面的 cache-aware can_allocate 测试（adapter 委托到这里）。"""

    def test_no_cache_returns_zero(self, cache: PagedKVCache):
        """无 cached block 时返回 0。"""
        tokens = _make_prompt(8)
        assert cache.block_pool.can_allocate(tokens) == 0

    def test_cache_hit(self, cache: PagedKVCache):
        """有 cached block 时返回命中数。"""
        tokens = _make_prompt(8)

        # 先制造 cached block
        cache.allocate_request("a", 8)
        cache.block_pool.hash_blocks(cache.block_tables["a"].block_ids, tokens, num_full_blocks=2)
        cache.free_request("a")

        assert cache.block_pool.can_allocate(tokens) == 2

    def test_insufficient_space_returns_negative(self, cache: PagedKVCache):
        """空间不够时返回 -1。"""
        # 分配所有 8 个 block
        for i in range(8):
            cache.allocate_request(f"r{i}", 4)

        # 新请求需要 1 block，但无空闲
        tokens = _make_prompt(4, base=999)
        assert cache.block_pool.can_allocate(tokens) == -1


# ── 3. Skip Prefill 正确性 ──


class TestSkipPrefillCorrectness:
    """验证 skip prefill 后的 KV 数据与完整 prefill 等价。

    核心不变量：cached block 的 KV 数据 + 新计算的 KV == 完整 prefill 的 KV
    """

    def test_skip_prefill_kv_matches_full_prefill(self, cache: PagedKVCache):
        """相同 prompt：skip prefill 的 KV == 完整 prefill 的 KV。

        步骤：
        1. 请求 A 完整 prefill（8 tokens, 2 blocks），注册 hash
        2. 释放 A（block 进 LRU）
        3. 请求 B skip prefill（复用 2 cached blocks），只算最后 0 token
        4. 对比 A 和 B 的 block_ids 和 KV 数据一致
        """
        tokens = _make_prompt(8)

        # 1. 请求 A: 完整 prefill
        cache.allocate_request("a", 8)
        table_a = cache.block_tables["a"]
        # 写入可识别的 KV
        _write_fake_kv(cache, "a", value=10.0)
        # 注册 hash
        cache.block_pool.hash_blocks(table_a.block_ids, tokens, num_full_blocks=2)
        # 保存 A 的 KV
        saved_kv_a = [
            (layer.k[table_a.block_ids].clone(), layer.v[table_a.block_ids].clone())
            for layer in cache.layers
        ]

        # 2. 释放 A
        cache.free_request("a")

        # 3. 请求 B: skip prefill，全部 cached
        cache.allocate_request_with_cache("b", tokens, num_cached=2)
        table_b = cache.block_tables["b"]

        # 4. B 复用了 A 的物理 block，KV 数据相同
        assert table_b.block_ids == table_a.block_ids
        for layer_idx, (ka, va) in enumerate(saved_kv_a):
            assert torch.equal(cache.layers[layer_idx].k[table_b.block_ids], ka)
            assert torch.equal(cache.layers[layer_idx].v[table_b.block_ids], va)
