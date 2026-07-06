"""BatchedKVCache + SlotManager 单元测试。

覆盖 L0 测试清单全部 9 项：
  1. cache shape [S, H_kv, L, D]
  2. dtype/device 继承 config
  3. allocate 顺序（从低 slot id 开始）
  4. 分配超过容量 → RuntimeError
  5. free 后可复用
  6. duplicate request_id → ValueError
  7. free 不存在的 request_id → ValueError
  8. seq_lens 初始化/重置（free 后 seq_len=0）
  9. occupied mask（allocate/free 后一致）
"""

import pytest
import torch

from inferlite.config import ModelConfig
from inferlite.model.batched_kv_cache import (
    BatchedKVCache,
    BatchedLayerKVCache,
    SlotManager,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tiny_config() -> ModelConfig:
    """小尺寸配置，避免实例化真实模型。"""
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=32,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def _make_cache(max_num_slots: int = 4, max_seq_len: int = 16) -> BatchedKVCache:
    """用 from_config 创建一个小 cache。"""
    config = _tiny_config()
    return BatchedKVCache.from_config(
        config=config,
        max_num_slots=max_num_slots,
        max_seq_len=max_seq_len,
        dtype=torch.float32,
        device="cpu",
    )


# ===========================================================================
# SlotManager 测试
# ===========================================================================


class TestSlotManager:
    """SlotManager 分配/释放/查询。"""

    def test_allocate_order(self):
        """L0-3: allocate 从低 slot id 开始。"""
        sm = SlotManager(3)
        assert sm.allocate("a") == 0
        assert sm.allocate("b") == 1
        assert sm.allocate("c") == 2

    def test_allocate_over_capacity(self):
        """L0-4: 超过容量抛 RuntimeError。"""
        sm = SlotManager(2)
        sm.allocate("a")
        sm.allocate("b")
        with pytest.raises(RuntimeError, match="no free slots"):
            sm.allocate("c")

    def test_free_and_reuse(self):
        """L0-5: free 后 slot 可再次 allocate。"""
        sm = SlotManager(2)
        sm.allocate("a")
        slot_b = sm.allocate("b")
        sm.free("b")
        # 释放后 is_free 应为 True
        assert sm.is_free(slot_b)
        # 再次分配应复用 slot_b
        slot_c = sm.allocate("c")
        assert slot_c == slot_b

    def test_duplicate_request_id(self):
        """L0-6: 重复 request_id 抛 ValueError。"""
        sm = SlotManager(2)
        sm.allocate("a")
        with pytest.raises(ValueError, match="already allocated"):
            sm.allocate("a")

    def test_free_not_found(self):
        """L0-7: free 不存在的 request_id 抛 ValueError。"""
        sm = SlotManager(2)
        sm.allocate("a")
        with pytest.raises(ValueError, match="not found"):
            sm.free("b")

    def test_is_free_initial(self):
        """初始时所有 slot 都空闲。"""
        sm = SlotManager(3)
        for i in range(3):
            assert sm.is_free(i)

    def test_is_free_after_allocate(self):
        """allocate 后 slot 不再空闲。"""
        sm = SlotManager(2)
        sm.allocate("a")
        assert not sm.is_free(0)
        assert sm.is_free(1)


# ===========================================================================
# BatchedLayerKVCache 测试
# ===========================================================================


class TestBatchedLayerKVCache:
    """BatchedLayerKVCache dataclass。"""

    def test_shape(self):
        """L0-1: k/v shape 为 [S, H_kv, L, D]。"""
        S, H_kv, L, D = 4, 2, 16, 8
        k = torch.zeros(S, H_kv, L, D)
        v = torch.zeros(S, H_kv, L, D)
        layer = BatchedLayerKVCache(k=k, v=v)
        assert layer.k.shape == (S, H_kv, L, D)
        assert layer.v.shape == (S, H_kv, L, D)


# ===========================================================================
# BatchedKVCache 测试
# ===========================================================================


class TestBatchedKVCache:
    """BatchedKVCache 构造、from_config、slot 管理。"""

    def test_from_config_shape(self):
        """L0-1: 每层 k/v shape 为 [max_num_slots, num_kv_heads, max_seq_len, head_dim]。"""
        cache = _make_cache(max_num_slots=4, max_seq_len=16)
        for layer in cache.layers:
            assert layer.k.shape == (4, 2, 16, 8)  # S=4, H_kv=2, L=16, D=8
            assert layer.v.shape == (4, 2, 16, 8)

    def test_from_config_num_layers(self):
        """from_config 创建的层数与 config 一致。"""
        cache = _make_cache()
        assert len(cache.layers) == _tiny_config().num_hidden_layers

    def test_dtype_device(self):
        """L0-2: dtype/device 与传入参数一致。"""
        cache = _make_cache()
        for layer in cache.layers:
            assert layer.k.dtype == torch.float32
            assert layer.k.device.type == "cpu"

    def test_seq_lens_init(self):
        """L0-8: seq_lens 初始化为全 0。"""
        cache = _make_cache(max_num_slots=4)
        assert (cache.seq_lens == 0).all()
        assert cache.seq_lens.shape == (4,)

    def test_occupied_init(self):
        """L0-9: occupied 初始化为全 False。"""
        cache = _make_cache(max_num_slots=4)
        assert not cache.occupied.any()
        assert cache.occupied.shape == (4,)

    def test_allocate_slot(self):
        """allocate_slot 分配 slot 并设置 occupied=True。"""
        cache = _make_cache(max_num_slots=3)
        slot_a = cache.allocate_slot("a")
        slot_b = cache.allocate_slot("b")
        assert slot_a == 0
        assert slot_b == 1
        assert cache.occupied[0].item() is True
        assert cache.occupied[1].item() is True
        assert cache.occupied[2].item() is False

    def test_free_slot_clears_metadata(self):
        """L0-8/9: free_slot 清 seq_lens 和 occupied。"""
        cache = _make_cache(max_num_slots=3)
        slot_a = cache.allocate_slot("a")
        # 模拟 prefill + decode 后 seq_lens 增长
        cache.seq_lens[slot_a] = 10
        assert cache.occupied[slot_a].item() is True
        assert cache.seq_lens[slot_a].item() == 10

        cache.free_slot("a")
        # free 后应全部清零
        assert cache.occupied[slot_a].item() is False
        assert cache.seq_lens[slot_a].item() == 0

    def test_free_slot_not_found(self):
        """free_slot 不存在的 request_id 抛 ValueError。"""
        cache = _make_cache()
        cache.allocate_slot("a")
        with pytest.raises(ValueError, match="not allocated"):
            cache.free_slot("b")

    def test_reset_slots(self):
        """reset_slots 清空所有 slot 状态。"""
        cache = _make_cache(max_num_slots=3)
        cache.allocate_slot("a")
        cache.allocate_slot("b")
        cache.seq_lens[0] = 5
        cache.seq_lens[1] = 12

        cache.reset_slots()
        assert (cache.seq_lens == 0).all()
        assert not cache.occupied.any()
        # SlotManager 也应该重置
        assert cache.slot_manager.is_free(0)
        assert cache.slot_manager.is_free(1)
        assert cache.slot_manager.is_free(2)

    def test_allocate_after_reset(self):
        """reset 后可以重新分配。"""
        cache = _make_cache(max_num_slots=2)
        cache.allocate_slot("a")
        cache.allocate_slot("b")
        cache.reset_slots()
        # 重新分配应从 slot 0 开始
        assert cache.allocate_slot("c") == 0
        assert cache.allocate_slot("d") == 1
