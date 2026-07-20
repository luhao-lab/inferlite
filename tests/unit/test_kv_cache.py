"""M2-T1 KVCache 数据结构单测。

测试目标：
- from_config 分配正确的 shape / dtype / device
- layers 数量等于 num_hidden_layers
- cur_len 初始为 0，reset() 后归零
- 越界写入触发 IndexError（切片赋值自然行为）
"""

import pytest
import torch

from inferlite.cache.kv_cache import KVCache, LayerKVCache
from inferlite.config import ModelConfig

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

BATCH_SIZE = 1
MAX_SEQ_LEN = 512


@pytest.fixture
def config() -> ModelConfig:
    return ModelConfig.qwen3_0_6b()


@pytest.fixture
def kv_cache(config: ModelConfig) -> KVCache:
    return KVCache.from_config(
        config,
        batch_size=BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        dtype=torch.float32,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Case 1: layers 数量等于 num_hidden_layers
# ---------------------------------------------------------------------------


def test_layer_count(config: ModelConfig, kv_cache: KVCache) -> None:
    assert len(kv_cache.layers) == config.num_hidden_layers


# ---------------------------------------------------------------------------
# Case 2: 每层 k/v 的 shape 正确
# ---------------------------------------------------------------------------


def test_layer_shape(config: ModelConfig, kv_cache: KVCache) -> None:
    expected = (BATCH_SIZE, config.num_key_value_heads, MAX_SEQ_LEN, config.head_dim)
    for layer in kv_cache.layers:
        assert layer.k.shape == expected, f"k shape mismatch: {layer.k.shape} != {expected}"
        assert layer.v.shape == expected, f"v shape mismatch: {layer.v.shape} != {expected}"


# ---------------------------------------------------------------------------
# Case 3: dtype 与构造参数一致
# ---------------------------------------------------------------------------


def test_dtype(kv_cache: KVCache) -> None:
    for layer in kv_cache.layers:
        assert layer.k.dtype == torch.float32
        assert layer.v.dtype == torch.float32


# ---------------------------------------------------------------------------
# Case 4: device 与构造参数一致
# ---------------------------------------------------------------------------


def test_device(kv_cache: KVCache) -> None:
    for layer in kv_cache.layers:
        assert layer.k.device.type == "cpu"
        assert layer.v.device.type == "cpu"


# ---------------------------------------------------------------------------
# Case 5: cur_len 初始为 0
# ---------------------------------------------------------------------------


def test_cur_len_initial(kv_cache: KVCache) -> None:
    assert kv_cache.cur_len == 0


# ---------------------------------------------------------------------------
# Case 6: reset() 后 cur_len 归零
# ---------------------------------------------------------------------------


def test_reset(kv_cache: KVCache) -> None:
    kv_cache.cur_len = 42
    kv_cache.reset()
    assert kv_cache.cur_len == 0


# ---------------------------------------------------------------------------
# Case 7: reset() 不清零 tensor（prefill 会覆盖，省时）
# ---------------------------------------------------------------------------


def test_reset_does_not_zero_tensor(kv_cache: KVCache) -> None:
    # 往第 0 层写入一些非零值
    kv_cache.layers[0].k[0, 0, 0, 0] = 3.14
    kv_cache.cur_len = 1
    kv_cache.reset()
    # tensor 数值应保留（不被清零）
    assert kv_cache.layers[0].k[0, 0, 0, 0] == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Case 8: 越界写入触发 IndexError
# ---------------------------------------------------------------------------


def test_out_of_bounds_write(kv_cache: KVCache) -> None:
    """向 max_seq_len 之外的位置写入应触发 IndexError。"""
    layer = kv_cache.layers[0]
    with pytest.raises(IndexError):
        # max_seq_len = 512，写第 513 位（索引 512）越界
        layer.k[:, :, MAX_SEQ_LEN, :] = 0.0


# ---------------------------------------------------------------------------
# Case 9: LayerKVCache 是 dataclass，支持直接构造
# ---------------------------------------------------------------------------


def test_layer_kvcache_direct_construct() -> None:
    k = torch.zeros(1, 4, 8, 64)
    v = torch.zeros_like(k)
    layer = LayerKVCache(k=k, v=v)
    assert layer.k is k
    assert layer.v is v


# ---------------------------------------------------------------------------
# Case 10: from_config 对 Qwen3-0.6B 标准参数的完整验证
# ---------------------------------------------------------------------------


def test_from_config_qwen3_0_6b() -> None:
    config = ModelConfig.qwen3_0_6b()
    cache = KVCache.from_config(
        config,
        batch_size=1,
        max_seq_len=1024,
        dtype=torch.float32,
        device="cpu",
    )
    # Qwen3-0.6B: 28 层, 8 KV heads, head_dim=128
    assert len(cache.layers) == 28
    assert cache.layers[0].k.shape == (1, 8, 1024, 128)
    assert cache.cur_len == 0
