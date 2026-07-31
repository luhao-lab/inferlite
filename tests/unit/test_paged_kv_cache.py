"""M4-T3 PagedKVCache 单元测试。

测试重点不是 attention 数学，而是分页地址转换：连续 batch K/V 如何依
BlockTable 的逻辑顺序 scatter 到不连续物理 slot，再 gather 回连续序列。
"""

import pytest
import torch

from inferlite.cache.paged_kv_cache import PagedKVCache
from inferlite.config import ModelConfig


@pytest.fixture
def config() -> ModelConfig:
    """足够小的两层配置，避免测试为 Qwen3 全尺寸 tensor 付费。"""
    return ModelConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=16,
        vocab_size=32,
        max_position_embeddings=32,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


@pytest.fixture
def cache(config: ModelConfig) -> PagedKVCache:
    return PagedKVCache.from_config(
        config, num_blocks=8, block_size=4, dtype=torch.float32, device="cpu"
    )


def _prefill_inputs(
    batch_size: int, t_max: int, value: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """创建 [B, n_kv=2, T, D=4] 的可识别 K/V；V 与 K 相差 100。"""
    k = torch.arange(batch_size * 2 * t_max * 4, dtype=torch.float32).reshape(
        batch_size, 2, t_max, 4
    )
    return k + value, k + value + 100.0


def test_from_config_builds_layers_and_pool(config: ModelConfig, cache: PagedKVCache) -> None:
    assert len(cache.layers) == config.num_hidden_layers
    assert cache.layers[0].k.shape == (8, 4, 2, 4)
    assert cache.layers[0].v.shape == (8, 4, 2, 4)
    assert cache.num_free_blocks == 8
    assert cache.can_allocate(5) is True


def test_allocate_and_free_restore_all_blocks(cache: PagedKVCache) -> None:
    cache.allocate_request("a", prompt_len=5)
    assert cache.block_tables["a"].block_ids == [0, 1]
    assert cache.seq_len_of("a") == 5
    assert cache.num_free_blocks == 6

    cache.free_request("a")
    assert cache.num_free_blocks == 8
    with pytest.raises(KeyError):
        cache.free_request("a")


def test_allocate_failure_rolls_back_blocks(cache: PagedKVCache) -> None:
    with pytest.raises(RuntimeError):
        cache.allocate_request("too-large", prompt_len=33)  # 需 9 块，池只有 8 块
    assert cache.num_free_blocks == 8
    with pytest.raises(KeyError):
        cache.seq_len_of("too-large")


def test_prefill_slot_mapping_matches_block_table(cache: PagedKVCache) -> None:
    cache.allocate_request("a", 5)  # block ids [0, 1] -> slots [0,1,2,3,4]
    cache.allocate_request("b", 2)  # block ids [2] -> slots [8,9]

    slots = cache._make_prefill_slot_mapping(["a", "b"], "cpu")
    assert slots.tolist() == [0, 1, 2, 3, 4, 8, 9]


def test_batch_prefill_scatter_and_single_gather_oracle(cache: PagedKVCache) -> None:
    cache.allocate_request("a", 5)
    cache.allocate_request("b", 2)
    k, v = _prefill_inputs(batch_size=2, t_max=5)
    # b 的后 3 个 token 是 batch padding；赋特殊值后也不得写入其物理 block。
    k[1, :, 2:, :] = -999.0
    v[1, :, 2:, :] = -999.0

    cache.write_prefill(0, ["a", "b"], k, v)
    a_k, a_v = cache.gather_kv_single(0, "a")
    b_k, b_v = cache.gather_kv_single(0, "b")

    assert torch.equal(a_k, k[0, :, :5, :])
    assert torch.equal(a_v, v[0, :, :5, :])
    assert torch.equal(b_k, k[1, :, :2, :])
    assert torch.equal(b_v, v[1, :, :2, :])
    assert not torch.any(b_k == -999.0)


def test_batch_gather_matches_single_oracle(cache: PagedKVCache) -> None:
    cache.allocate_request("a", 5)
    cache.allocate_request("b", 2)
    k, v = _prefill_inputs(2, 5)
    cache.write_prefill(0, ["a", "b"], k, v)

    batch_k, batch_v, valid_lens = cache.gather_kv(0, ["a", "b"])
    assert batch_k.shape == (2, 2, 8, 4)  # max 2 blocks * block_size 4
    assert valid_lens.tolist() == [5, 2]
    for index, request_id in enumerate(["a", "b"]):
        single_k, single_v = cache.gather_kv_single(0, request_id)
        length = valid_lens[index].item()
        assert torch.equal(batch_k[index, :, :length, :], single_k)
        assert torch.equal(batch_v[index, :, :length, :], single_v)


def test_batch_decode_writes_boundary_and_non_boundary_requests(cache: PagedKVCache) -> None:
    cache.allocate_request("a", 4)  # 正好写满一个 block
    cache.allocate_request("b", 3)  # 同一块还剩一个位置
    k, v = _prefill_inputs(2, 4)
    cache.write_prefill(0, ["a", "b"], k, v)

    cache.append_token("a")  # a 新拿 block 2，写 offset 0
    cache.append_token("b")  # b 仍在 block 1，写 offset 3
    decode_k = torch.full((2, 2, 1, 4), 7.0)
    decode_v = torch.full((2, 2, 1, 4), 9.0)
    cache.write_decode(0, ["a", "b"], decode_k, decode_v)

    assert cache.block_tables["a"].block_ids == [0, 2]
    assert cache.block_tables["b"].block_ids == [1]
    a_k, a_v = cache.gather_kv_single(0, "a")
    b_k, b_v = cache.gather_kv_single(0, "b")
    assert torch.equal(a_k[:, -1, :], decode_k[0, :, 0, :])
    assert torch.equal(a_v[:, -1, :], decode_v[0, :, 0, :])
    assert torch.equal(b_k[:, -1, :], decode_k[1, :, 0, :])
    assert torch.equal(b_v[:, -1, :], decode_v[1, :, 0, :])


def test_batch_write_rejects_bad_contracts(cache: PagedKVCache) -> None:
    cache.allocate_request("a", 2)
    k, v = _prefill_inputs(1, 2)

    with pytest.raises(ValueError, match="duplicates"):
        cache.write_prefill(0, ["a", "a"], k.repeat(2, 1, 1, 1), v.repeat(2, 1, 1, 1))
    with pytest.raises(ValueError, match="batch size"):
        cache.write_prefill(0, ["a", "missing"], k, v)
    with pytest.raises(ValueError, match="T_max"):
        cache.write_prefill(0, ["a"], k[:, :, :1, :], v[:, :, :1, :])
    with pytest.raises(ValueError, match="shape"):
        cache.write_decode(0, ["a"], k, v)
    with pytest.raises(IndexError):
        cache.write_prefill(99, ["a"], k, v)


def test_unknown_request_interfaces_raise_key_error(cache: PagedKVCache) -> None:
    k, v = _prefill_inputs(1, 1)
    with pytest.raises(KeyError):
        cache.append_token("missing")
    with pytest.raises(KeyError):
        cache.write_prefill(0, ["missing"], k, v)
    with pytest.raises(KeyError):
        cache.gather_kv(0, ["missing"])
    with pytest.raises(ValueError, match="empty"):
        cache.gather_kv(0, [])
