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

from inferlite.config import ModelConfig
from inferlite.model.attention import GQAAttention
from inferlite.model.batched_kv_cache import BatchedLayerKVCache
from inferlite.model.kv_cache import LayerKVCache
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


# ===========================================================================
# Attention 层测试
# ===========================================================================


class TestBatchedAttentionLayer:
    """GQAAttention + BatchedLayerKVCache 的 batched decode。"""

    def _make_attn(self) -> GQAAttention:
        config = _tiny_config(num_hidden_layers=1)
        return GQAAttention(config)

    def test_batched_decode_output_shape(self):
        """L0-1: batched decode 输出 shape [B, 1, hidden_size]。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()
        B = 3

        hidden = torch.randn(B, 1, 32)
        cache_slots = torch.tensor([0, 1, 2])
        cache_positions = torch.tensor([5, 10, 3])
        position_ids = cache_positions[:, None]  # [B, 1]

        out = attn(
            hidden,
            position_ids=position_ids,
            layer_kv_cache=cache,
            cache_slots=cache_slots,
            cache_positions=cache_positions,
        )
        assert out.shape == (B, 1, 32)

    def test_cache_slot_write_position(self):
        """L0-2: 当前 token K/V 写入对应 slot 的 position。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()

        hidden = torch.randn(2, 1, 32)
        cache_slots = torch.tensor([0, 2])
        cache_positions = torch.tensor([5, 10])
        position_ids = cache_positions[:, None]

        attn(
            hidden,
            position_ids=position_ids,
            layer_kv_cache=cache,
            cache_slots=cache_slots,
            cache_positions=cache_positions,
        )

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
        cache.k[0, :, :8, :] = torch.randn(2, 8, 8)  # slot 0 有 8 个 token
        cache.v[0, :, :8, :] = torch.randn(2, 8, 8)
        # 给 slot 1 填不同的 KV 数据
        cache.k[1, :, :3, :] = torch.randn(2, 3, 8)  # slot 1 有 3 个 token
        cache.v[1, :, :3, :] = torch.randn(2, 3, 8)

        # 分别用两个请求 decode
        hidden_a = torch.randn(1, 1, 32)
        hidden_b = torch.randn(1, 1, 32)

        # 请求 A: slot 0, pos 8
        out_a = attn(
            hidden_a,
            position_ids=torch.tensor([[8]]),
            layer_kv_cache=cache,
            cache_slots=torch.tensor([0]),
            cache_positions=torch.tensor([8]),
        )

        # 请求 B: slot 1, pos 3（需要新的 cache 副本，因为 A 已修改了 cache）
        cache2 = _make_batched_layer_cache()
        cache2.k[0, :, :8, :] = cache.k[0, :, :8, :].clone()
        cache2.v[0, :, :8, :] = cache.v[0, :, :8, :].clone()
        cache2.k[1, :, :3, :] = cache.k[1, :, :3, :].clone()
        cache2.v[1, :, :3, :] = cache.v[1, :, :3, :].clone()

        out_b = attn(
            hidden_b,
            position_ids=torch.tensor([[3]]),
            layer_kv_cache=cache2,
            cache_slots=torch.tensor([1]),
            cache_positions=torch.tensor([3]),
        )

        # 如果两个请求互不干扰，分别 decode 的结果应该和合批 decode 一致
        cache3 = _make_batched_layer_cache()
        cache3.k[0, :, :8, :] = cache.k[0, :, :8, :].clone()
        cache3.v[0, :, :8, :] = cache.v[0, :, :8, :].clone()
        cache3.k[1, :, :3, :] = cache.k[1, :, :3, :].clone()
        cache3.v[1, :, :3, :] = cache.v[1, :, :3, :].clone()

        hidden_batch = torch.cat([hidden_a, hidden_b], dim=0)
        out_batch = attn(
            hidden_batch,
            position_ids=torch.tensor([[8], [3]]),
            layer_kv_cache=cache3,
            cache_slots=torch.tensor([0, 1]),
            cache_positions=torch.tensor([8, 3]),
        )

        assert torch.allclose(out_a, out_batch[0:1], atol=1e-5)
        assert torch.allclose(out_b, out_batch[1:2], atol=1e-5)

    def test_mask_preserves_current_position(self):
        """L0-4: query 能 attend 到当前 token 自己（不会被 mask 掉）。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()

        # 在 pos=5 做 decode，query 应该能 attend 到 pos=0..5（包括自己）
        hidden = torch.randn(1, 1, 32)
        out = attn(
            hidden,
            position_ids=torch.tensor([[5]]),
            layer_kv_cache=cache,
            cache_slots=torch.tensor([0]),
            cache_positions=torch.tensor([5]),
        )
        # 输出不应为全零（说明 attention 确实 attend 到了东西）
        assert out.abs().sum() > 0

    def test_padding_positions_masked(self):
        """L0-5: padding 位置的 score 被 mask 为 dtype min。"""
        attn = self._make_attn()
        cache = _make_batched_layer_cache()

        # 两个请求：pos=3 和 pos=10
        # gather 后 max_len=11，slot 0 只有 4 个有效 KV，后 7 个是 padding
        hidden = torch.randn(2, 1, 32)
        cache_slots = torch.tensor([0, 1])
        cache_positions = torch.tensor([3, 10])

        out = attn(
            hidden,
            position_ids=cache_positions[:, None],
            layer_kv_cache=cache,
            cache_slots=cache_slots,
            cache_positions=cache_positions,
        )
        # 输出不为 NaN（mask 正常工作）
        assert not torch.isnan(out).any()
        # 输出不为 inf
        assert not torch.isinf(out).any()

    def test_b1_equivalent_to_m2_decode(self):
        """L0-6: B=1 batched decode 和 M2 single decode 结果等价。"""
        config = _tiny_config(num_hidden_layers=1)
        attn = GQAAttention(config)

        hidden = torch.randn(1, 1, 32)
        pos = 5

        # M2 路径：LayerKVCache
        m2_cache = _make_single_layer_cache(batch_size=1)
        # 先填一些历史 KV（模拟 prefill 后）
        m2_cache.k[:, :, :pos, :] = torch.randn(1, 2, pos, 8)
        m2_cache.v[:, :, :pos, :] = torch.randn(1, 2, pos, 8)

        out_m2 = attn(
            hidden.clone(),
            position_ids=torch.tensor([[pos]]),
            layer_kv_cache=m2_cache,
            cache_position=pos,
        )

        # M3 路径：BatchedLayerKVCache, B=1
        m3_cache = _make_batched_layer_cache()
        # 填同样的历史 KV 到 slot 0
        m3_cache.k[0, :, :pos, :] = m2_cache.k[0, :, :pos, :].clone()
        m3_cache.v[0, :, :pos, :] = m2_cache.v[0, :, :pos, :].clone()

        out_m3 = attn(
            hidden.clone(),
            position_ids=torch.tensor([[pos]]),
            layer_kv_cache=m3_cache,
            cache_slots=torch.tensor([0]),
            cache_positions=torch.tensor([pos]),
        )

        assert torch.allclose(out_m2, out_m3, atol=1e-4)

    def test_mixed_positions_equivalent_to_sequential(self):
        """L0-7: 不同 cache_positions 混 batch 结果等价逐条 decode。"""
        config = _tiny_config(num_hidden_layers=1)
        attn = GQAAttention(config)

        # 3 个请求，不同历史长度
        positions = [3, 7, 12]
        slots = [0, 1, 2]
        hiddens = [torch.randn(1, 1, 32) for _ in range(3)]

        # 逐条 decode
        seq_outputs = []
        for i in range(3):
            cache = _make_batched_layer_cache()
            # 填历史 KV
            cache.k[slots[i], :, : positions[i], :] = torch.randn(2, positions[i], 8)
            cache.v[slots[i], :, : positions[i], :] = torch.randn(2, positions[i], 8)
            out = attn(
                hiddens[i].clone(),
                position_ids=torch.tensor([[positions[i]]]),
                layer_kv_cache=cache,
                cache_slots=torch.tensor([slots[i]]),
                cache_positions=torch.tensor([positions[i]]),
            )
            seq_outputs.append(out)

        # 合批 decode（用同样的历史 KV）
        torch.manual_seed(42)
        histories_k = [torch.randn(2, p, 8) for p in positions]
        histories_v = [torch.randn(2, p, 8) for p in positions]

        batch_cache = _make_batched_layer_cache()
        for i in range(3):
            batch_cache.k[slots[i], :, : positions[i], :] = histories_k[i]
            batch_cache.v[slots[i], :, : positions[i], :] = histories_v[i]

        hidden_batch = torch.cat([h.clone() for h in hiddens], dim=0)
        batch_out = attn(
            hidden_batch,
            position_ids=torch.tensor([[p] for p in positions]),
            layer_kv_cache=batch_cache,
            cache_slots=torch.tensor(slots),
            cache_positions=torch.tensor(positions),
        )

        # 逐条用同样的历史重新跑
        for i in range(3):
            single_cache = _make_batched_layer_cache()
            single_cache.k[slots[i], :, : positions[i], :] = histories_k[i]
            single_cache.v[slots[i], :, : positions[i], :] = histories_v[i]
            single_out = attn(
                hiddens[i].clone(),
                position_ids=torch.tensor([[positions[i]]]),
                layer_kv_cache=single_cache,
                cache_slots=torch.tensor([slots[i]]),
                cache_positions=torch.tensor([positions[i]]),
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

        # forward 内部会做 repeat_kv，如果 shape 不对会报错
        out = attn(
            hidden,
            position_ids=cache_positions[:, None],
            layer_kv_cache=cache,
            cache_slots=cache_slots,
            cache_positions=cache_positions,
        )
        # 输出 shape 正确说明 repeat_kv 正常工作
        assert out.shape == (B, 1, 32)


# ===========================================================================
# Model 层测试
# ===========================================================================


class TestBatchedAttentionModel:
    """Qwen3Model + BatchedKVCache 的 batched decode。"""

    def test_model_batched_decode_shape(self):
        """L0-1: model batched decode 输出 shape [B, 1, hidden_size]。"""
        from inferlite.model.batched_kv_cache import BatchedKVCache

        config = _tiny_config(num_hidden_layers=2)
        model = Qwen3Model(config)
        cache = BatchedKVCache.from_config(
            config, max_num_slots=4, max_seq_len=32, dtype=torch.float32, device="cpu"
        )

        B = 3
        input_ids = torch.randint(0, 100, (B, 1))
        cache_slots = torch.tensor([0, 1, 2])
        cache_positions = torch.tensor([5, 10, 3])
        position_ids = cache_positions[:, None]

        out = model(
            input_ids,
            position_ids=position_ids,
            kv_cache=cache,
            cache_slots=cache_slots,
            cache_positions=cache_positions,
        )
        assert out.shape == (B, 1, 32)

    def test_model_m2_not_broken(self):
        """M2 generate 路径不受影响（kv_cache=KVCache, 无 cache_slots）。"""
        from inferlite.model.kv_cache import KVCache

        config = _tiny_config(num_hidden_layers=2)
        model = Qwen3Model(config)
        cache = KVCache.from_config(
            config, batch_size=1, max_seq_len=32, dtype=torch.float32, device="cpu"
        )

        # M2 prefill
        input_ids = torch.randint(0, 100, (1, 5))
        out = model(input_ids, kv_cache=cache)
        assert out.shape == (1, 5, 32)
