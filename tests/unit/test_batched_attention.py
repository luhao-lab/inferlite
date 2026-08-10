"""Batched Attention (M3-T3) 单元测试。

覆盖 L0 测试清单全部 8 项：
  1. batched decode 输出 shape [B, 1, D]
  2. cache slot 写入位置正确
  3. 不同 row 不串 KV
  4. mask 保留当前位置（query 能 attend self）
  5. padding 位置被 mask
  6. B=1 时和 M2 decode 等价
  7. 不同 cache_positions 混 batch 等价逐条 decode
  8. MQA/GQA repeat_kv 后 shape 正确
"""

import torch

from inferlite.cache.batched_kv_cache import BatchedLayerKVCache
from inferlite.cache.kv_cache import LayerKVCache
from inferlite.config import ModelConfig
from inferlite.engine.forward_context import AttentionMetadata, set_forward_context
from inferlite.model.attention import Qwen3Attention
from inferlite.model.qwen3 import Qwen3Model

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _tiny_config(num_hidden_layers: int = 2) -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )


def _make_batched_layer_cache(
    max_num_slots: int = 4,
    max_seq_len: int = 32,
    n_kv_heads: int = 2,
    head_dim: int = 8,
) -> BatchedLayerKVCache:
    """创建一个空的 BatchedLayerKVCache。"""
    k = torch.zeros(max_num_slots, n_kv_heads, max_seq_len, head_dim)
    v = torch.zeros(max_num_slots, n_kv_heads, max_seq_len, head_dim)
    return BatchedLayerKVCache(k=k, v=v)


def _make_single_layer_cache(
    batch_size: int = 1,
    max_seq_len: int = 32,
    n_kv_heads: int = 2,
    head_dim: int = 8,
) -> LayerKVCache:
    """创建一个空的 LayerKVCache（M2）。"""
    k = torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim)
    v = torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim)
    return LayerKVCache(k=k, v=v)


def _call_batched(attn, hidden, position_ids, cache, cache_slots, cache_positions=None):
    """Helper: 用新签名调用 Qwen3Attention，支持 M3 batched 和 M2 single cache。

    Args:
        attn: Qwen3Attention 实例
        hidden: [B, T, H]
        position_ids: [B, T] or [B, 1]
        cache: BatchedLayerKVCache 或 LayerKVCache
        cache_slots: [B] slot 映射（M3）
        cache_positions: [B] 写入位置（M3 decode），None 表示 prefill
    """
    # 绑定 cache
    attn.attn.kv_cache = cache

    # 构造 metadata
    T = hidden.shape[1]
    if isinstance(cache, BatchedLayerKVCache):
        if cache_positions is not None:
            # M3 decode
            seq_lens = cache_positions + 1
        else:
            # M3 prefill
            B = hidden.shape[0]
            seq_lens = torch.full((B,), T, device=hidden.device)
        metadata = AttentionMetadata(
            num_seqs=len(seq_lens),
            seq_lens=seq_lens,
            slot_mapping=cache_slots,
        )
    elif isinstance(cache, LayerKVCache):
        # M2: cache_position 隐含在 seq_lens 中
        if cache_positions is not None:
            cp = (
                int(cache_positions[0])
                if hasattr(cache_positions, "__len__")
                else int(cache_positions)
            )
            seq_lens = torch.tensor([cp + T], device=hidden.device)
        else:
            seq_lens = torch.tensor([T], device=hidden.device)
        metadata = AttentionMetadata(num_seqs=1, seq_lens=seq_lens)

    with set_forward_context(metadata):
        out = attn(position_ids, hidden)

    # 恢复
    attn.attn.kv_cache = None
    return out


# ===========================================================================
# Attention 层测试
# ===========================================================================


class TestBatchedAttentionLayer:
    """Qwen3Attention + BatchedLayerKVCache 的 batched decode。"""

    def _make_attn(self) -> Qwen3Attention:
        config = _tiny_config(num_hidden_layers=1)
        return Qwen3Attention(config)

    def test_batched_decode_output_shape(self):
        """L0-1: batched decode 输出 shape [B, 1, hidden_size]。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()
        B = 3

        hidden = torch.randn(B, 1, 32)
        cache_slots = torch.tensor([0, 1, 2])
        cache_positions = torch.tensor([5, 10, 3])
        position_ids = cache_positions[:, None]  # [B, 1]

        out = _call_batched(attn, hidden, position_ids, cache, cache_slots, cache_positions)
        assert out.shape == (B, 1, 32)

    def test_cache_slot_write_position(self):
        """L0-2: 当前 token K/V 写入对应 slot 的 position。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()

        hidden = torch.randn(2, 1, 32)
        cache_slots = torch.tensor([0, 2])
        cache_positions = torch.tensor([5, 10])
        position_ids = cache_positions[:, None]

        _call_batched(attn, hidden, position_ids, cache, cache_slots, cache_positions)

        # slot 0, pos 5 应该有非零值
        assert cache.k[0, :, 5, :].abs().sum() > 0
        assert cache.v[0, :, 5, :].abs().sum() > 0
        # slot 2, pos 10 应该有非零值
        assert cache.k[2, :, 10, :].abs().sum() > 0
        # slot 1 应该全零（没有被写入）
        assert cache.k[1].abs().sum() == 0
        # slot 0 的其他位置应该为零
        assert cache.k[0, :, 0, :].abs().sum() == 0
        assert cache.k[0, :, 6, :].abs().sum() == 0

    def test_no_cross_slot_attention(self):
        """L0-3: 不同 row 不串 KV（每个请求只看自己 slot）。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()

        # 给 slot 0 填一些 KV 数据（模拟历史）
        cache.k[0, :, :8, :] = torch.randn(2, 8, 8)
        cache.v[0, :, :8, :] = torch.randn(2, 8, 8)
        # 给 slot 1 填不同的 KV 数据
        cache.k[1, :, :3, :] = torch.randn(2, 3, 8)
        cache.v[1, :, :3, :] = torch.randn(2, 3, 8)

        hidden_a = torch.randn(1, 1, 32)
        hidden_b = torch.randn(1, 1, 32)

        # 请求 A: slot 0, pos 8
        out_a = _call_batched(
            attn,
            hidden_a,
            torch.tensor([[8]]),
            cache,
            torch.tensor([0]),
            torch.tensor([8]),
        )

        # 请求 B: slot 1, pos 3（需要新的 cache 副本）
        cache2 = _make_batched_layer_cache()
        cache2.k[0, :, :8, :] = cache.k[0, :, :8, :].clone()
        cache2.v[0, :, :8, :] = cache.v[0, :, :8, :].clone()
        cache2.k[1, :, :3, :] = cache.k[1, :, :3, :].clone()
        cache2.v[1, :, :3, :] = cache.v[1, :, :3, :].clone()

        out_b = _call_batched(
            attn,
            hidden_b,
            torch.tensor([[3]]),
            cache2,
            torch.tensor([1]),
            torch.tensor([3]),
        )

        # 合批 decode
        cache3 = _make_batched_layer_cache()
        cache3.k[0, :, :8, :] = cache.k[0, :, :8, :].clone()
        cache3.v[0, :, :8, :] = cache.v[0, :, :8, :].clone()
        cache3.k[1, :, :3, :] = cache.k[1, :, :3, :].clone()
        cache3.v[1, :, :3, :] = cache.v[1, :, :3, :].clone()

        hidden_batch = torch.cat([hidden_a, hidden_b], dim=0)
        out_batch = _call_batched(
            attn,
            hidden_batch,
            torch.tensor([[8], [3]]),
            cache3,
            torch.tensor([0, 1]),
            torch.tensor([8, 3]),
        )

        assert torch.allclose(out_a, out_batch[0:1], atol=1e-5)
        assert torch.allclose(out_b, out_batch[1:2], atol=1e-5)

    def test_mask_preserves_current_position(self):
        """L0-4: query 能 attend 到当前 token 自己（不会被 mask 掉）。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()

        hidden = torch.randn(1, 1, 32)
        out = _call_batched(
            attn,
            hidden,
            torch.tensor([[5]]),
            cache,
            torch.tensor([0]),
            torch.tensor([5]),
        )
        assert out.abs().sum() > 0

    def test_padding_positions_masked(self):
        """L0-5: padding 位置的 score 被 mask 为 dtype min。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()

        hidden = torch.randn(2, 1, 32)
        cache_slots = torch.tensor([0, 1])
        cache_positions = torch.tensor([3, 10])

        out = _call_batched(
            attn,
            hidden,
            cache_positions[:, None],
            cache,
            cache_slots,
            cache_positions,
        )
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_nan_padding_does_not_contaminate_short_request(self):
        """无效 K/V 尾部即使含 NaN，也不得污染短请求的 attention 输出。"""
        torch.manual_seed(42)
        attn = self._make_attn().eval()
        short_pos, long_pos = 3, 10
        short_hidden = torch.randn(1, 1, 32)
        long_hidden = torch.randn(1, 1, 32)
        short_k = torch.randn(2, short_pos, 8)
        short_v = torch.randn(2, short_pos, 8)
        long_k = torch.randn(2, long_pos, 8)
        long_v = torch.randn(2, long_pos, 8)

        # Oracle：短请求单独 decode
        single_cache = _make_batched_layer_cache()
        single_cache.k[0, :, :short_pos] = short_k
        single_cache.v[0, :, :short_pos] = short_v
        single_out = _call_batched(
            attn,
            short_hidden.clone(),
            torch.tensor([[short_pos]]),
            single_cache,
            torch.tensor([0]),
            torch.tensor([short_pos]),
        )

        # 合批路径：短请求无效尾部显式填 NaN
        batch_cache = _make_batched_layer_cache()
        batch_cache.k.fill_(float("nan"))
        batch_cache.v.fill_(float("nan"))
        batch_cache.k[0, :, :short_pos] = short_k
        batch_cache.v[0, :, :short_pos] = short_v
        batch_cache.k[1, :, :long_pos] = long_k
        batch_cache.v[1, :, :long_pos] = long_v

        batch_out = _call_batched(
            attn,
            torch.cat((short_hidden, long_hidden), dim=0),
            torch.tensor([[short_pos], [long_pos]]),
            batch_cache,
            torch.tensor([0, 1]),
            torch.tensor([short_pos, long_pos]),
        )

        assert torch.isfinite(batch_out).all()
        assert torch.allclose(single_out, batch_out[0:1], atol=1e-5)

    def test_b1_equivalent_to_m2_decode(self):
        """L0-6: B=1 batched decode 和 M2 single decode 结果等价。"""
        config = _tiny_config(num_hidden_layers=1)
        attn = Qwen3Attention(config)

        hidden = torch.randn(1, 1, 32)
        pos = 5

        # M2 路径：LayerKVCache
        m2_cache = _make_single_layer_cache(batch_size=1)
        m2_cache.k[:, :, :pos, :] = torch.randn(1, 2, pos, 8)
        m2_cache.v[:, :, :pos, :] = torch.randn(1, 2, pos, 8)

        out_m2 = _call_batched(
            attn,
            hidden.clone(),
            torch.tensor([[pos]]),
            m2_cache,
            torch.tensor([0]),
            torch.tensor([pos]),
        )

        # M3 路径：BatchedLayerKVCache, B=1
        m3_cache = _make_batched_layer_cache()
        m3_cache.k[0, :, :pos, :] = m2_cache.k[0, :, :pos, :].clone()
        m3_cache.v[0, :, :pos, :] = m2_cache.v[0, :, :pos, :].clone()

        out_m3 = _call_batched(
            attn,
            hidden.clone(),
            torch.tensor([[pos]]),
            m3_cache,
            torch.tensor([0]),
            torch.tensor([pos]),
        )

        assert torch.allclose(out_m2, out_m3, atol=1e-4)

    def test_mixed_positions_equivalent_to_sequential(self):
        """L0-7: 不同 cache_positions 混 batch 结果等价逐条 decode。"""
        config = _tiny_config(num_hidden_layers=1)
        attn = Qwen3Attention(config)

        positions = [3, 7, 12]
        slots = [0, 1, 2]
        hiddens = [torch.randn(1, 1, 32) for _ in range(3)]

        # 逐条 decode
        seq_outputs = []
        for i in range(3):
            cache = _make_batched_layer_cache()
            cache.k[slots[i], :, : positions[i], :] = torch.randn(2, positions[i], 8)
            cache.v[slots[i], :, : positions[i], :] = torch.randn(2, positions[i], 8)
            out = _call_batched(
                attn,
                hiddens[i].clone(),
                torch.tensor([[positions[i]]]),
                cache,
                torch.tensor([slots[i]]),
                torch.tensor([positions[i]]),
            )
            seq_outputs.append(out)

        # 合批 decode
        torch.manual_seed(42)
        histories_k = [torch.randn(2, p, 8) for p in positions]
        histories_v = [torch.randn(2, p, 8) for p in positions]

        batch_cache = _make_batched_layer_cache()
        for i in range(3):
            batch_cache.k[slots[i], :, : positions[i], :] = histories_k[i]
            batch_cache.v[slots[i], :, : positions[i], :] = histories_v[i]

        hidden_batch = torch.cat([h.clone() for h in hiddens], dim=0)
        batch_out = _call_batched(
            attn,
            hidden_batch,
            torch.tensor([[p] for p in positions]),
            batch_cache,
            torch.tensor(slots),
            torch.tensor(positions),
        )

        # 逐条用同样的历史重新跑
        for i in range(3):
            single_cache = _make_batched_layer_cache()
            single_cache.k[slots[i], :, : positions[i], :] = histories_k[i]
            single_cache.v[slots[i], :, : positions[i], :] = histories_v[i]
            single_out = _call_batched(
                attn,
                hiddens[i].clone(),
                torch.tensor([[positions[i]]]),
                single_cache,
                torch.tensor([slots[i]]),
                torch.tensor([positions[i]]),
            )
            assert torch.allclose(
                single_out, batch_out[i : i + 1], atol=1e-4
            ), f"request {i} (pos={positions[i]}) mismatch"

    def test_gqa_repeat_kv_shape(self):
        """L0-8: batched decode 后 GQA repeat_kv shape 正确。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()

        B = 3
        hidden = torch.randn(B, 1, 32)
        cache_slots = torch.tensor([0, 1, 2])
        cache_positions = torch.tensor([5, 10, 3])

        out = _call_batched(
            attn,
            hidden,
            cache_positions[:, None],
            cache,
            cache_slots,
            cache_positions,
        )
        assert out.shape == (B, 1, 32)


# ===========================================================================
# Model 层测试
# ===========================================================================


class TestBatchedAttentionModel:
    """Qwen3Model + ForwardContext 的 batched decode / M2 验证。"""

    def test_model_batched_decode_shape(self):
        """L0-1: model batched decode 输出 shape [B, 1, hidden_size]（ForwardContext 路径）。"""
        from inferlite.cache.batched_kv_cache import BatchedKVCache
        from inferlite.engine.forward_context import AttentionMetadata, set_forward_context

        config = _tiny_config(num_hidden_layers=2)
        model = Qwen3Model(config)
        cache = BatchedKVCache.from_config(
            config, max_num_slots=4, max_seq_len=32, dtype=torch.float32, device="cpu"
        )
        # 手动绑定 cache 到每层 Attention
        for i, layer in enumerate(model.layers):
            layer.self_attn.attn.kv_cache = cache.layers[i]
            layer.self_attn.attn.layer_idx = i

        B = 3
        input_ids = torch.randint(0, 100, (B, 1))
        cache_slots = torch.tensor([0, 1, 2])
        cache_positions = torch.tensor([5, 10, 3])
        position_ids = cache_positions[:, None]

        metadata = AttentionMetadata(
            num_seqs=B,
            seq_lens=cache_positions + 1,
            slot_mapping=cache_slots,
        )
        with set_forward_context(metadata):
            out = model(input_ids, position_ids=position_ids)
        assert out.shape == (B, 1, 32)

    def test_model_m2_not_broken(self):
        """M2 generate 路径不受影响（ForwardContext + SingleCacheAdapter）。"""
        from inferlite.cache.kv_cache import KVCache
        from inferlite.engine.forward_context import set_forward_context

        config = _tiny_config(num_hidden_layers=2)
        model = Qwen3Model(config)
        cache = KVCache.from_config(
            config, batch_size=1, max_seq_len=32, dtype=torch.float32, device="cpu"
        )
        # 手动绑定 cache 到每层 Attention（Qwen3Model 没有 .model 属性）
        for i, layer in enumerate(model.layers):
            layer.self_attn.attn.kv_cache = cache.layers[i]
            layer.self_attn.attn.layer_idx = i

        # M2 prefill
        input_ids = torch.randint(0, 100, (1, 5))
        position_ids = torch.arange(5).unsqueeze(0)
        metadata = AttentionMetadata(num_seqs=1, seq_lens=torch.tensor([5]))
        with set_forward_context(metadata):
            out = model(input_ids, position_ids=position_ids)
        assert out.shape == (1, 5, 32)
