"""
Unit tests for inferlite.model.layers RoPE utilities.

T3 目标：手写 Qwen3 的 RoPE 基础算子，并与 transformers.Qwen3RotaryEmbedding
和 apply_rotary_pos_emb 数值对齐。

运行：
  uv run pytest tests/unit/test_rope.py -q
"""

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb as ref_apply_rotary_pos_emb,
)
from transformers.models.qwen3.modeling_qwen3 import (
    rotate_half as ref_rotate_half,
)

from inferlite.model.layers import RotaryEmbedding, apply_rotary_pos_emb, rotate_half


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotate_half_vs_transformers(dtype):
    """rotate_half 必须与 transformers 的前半/后半切分实现一致。"""
    x = torch.arange(2 * 3 * 8, dtype=dtype).reshape(2, 3, 8)

    y_ref = ref_rotate_half(x)
    y_mine = rotate_half(x)

    assert y_mine.shape == x.shape
    assert torch.equal(y_mine, y_ref)


def test_rotate_half_known_example():
    """用手算例子锁死方向：不是 even/odd interleave，也不是 [-x1, x1]。"""
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    y = rotate_half(x)

    assert torch.equal(y, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_embedding_cos_sin_vs_qwen3_rotary_embedding(dtype):
    """RotaryEmbedding 生成的 cos/sin 与 transformers.Qwen3RotaryEmbedding 对齐。"""
    rope_theta = 1_000_000.0
    cfg = Qwen3Config(
        hidden_size=16,
        num_attention_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
    )
    x = torch.randn(2, 2, 5, cfg.head_dim, dtype=dtype)
    position_ids = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
        ],
        dtype=torch.long,
    )

    ref = Qwen3RotaryEmbedding(cfg).to(dtype).eval()
    mine = (
        RotaryEmbedding(
            head_dim=cfg.head_dim,
            rope_theta=rope_theta,
        )
        .to(dtype)
        .eval()
    )

    with torch.no_grad():
        cos_ref, sin_ref = ref(x, position_ids)
        cos_mine, sin_mine = mine(x, position_ids)

    assert cos_mine.shape == (2, 5, cfg.head_dim)
    assert sin_mine.shape == (2, 5, cfg.head_dim)
    atol = 1e-6 if dtype == torch.float32 else 5e-3
    assert torch.allclose(cos_mine, cos_ref, atol=atol, rtol=1e-4)
    assert torch.allclose(sin_mine, sin_ref, atol=atol, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_apply_rotary_pos_emb_vs_transformers(dtype):
    """apply_rotary_pos_emb 对 [B, heads, T, D] 的 q/k 做广播旋转。"""
    torch.manual_seed(0)
    batch_size = 2
    num_heads = 3
    seq_len = 5
    head_dim = 8

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    cos = torch.randn(batch_size, seq_len, head_dim, dtype=dtype)
    sin = torch.randn(batch_size, seq_len, head_dim, dtype=dtype)

    q_ref, k_ref = ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    q_mine, k_mine = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

    assert q_mine.shape == q.shape
    assert k_mine.shape == k.shape
    assert torch.allclose(q_mine, q_ref, atol=0, rtol=0)
    assert torch.allclose(k_mine, k_ref, atol=0, rtol=0)


def test_apply_rotary_pos_emb_supports_seq_major_broadcast():
    """unsqueeze_dim=2 时支持 [B, T, heads, D]，与 transformers 行为一致。"""
    torch.manual_seed(0)
    q = torch.randn(2, 5, 3, 8)
    k = torch.randn(2, 5, 3, 8)
    cos = torch.randn(2, 5, 8)
    sin = torch.randn(2, 5, 8)

    q_ref, k_ref = ref_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
    q_mine, k_mine = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)

    assert q_mine.shape == q.shape
    assert k_mine.shape == k.shape
    assert torch.allclose(q_mine, q_ref, atol=0, rtol=0)
    assert torch.allclose(k_mine, k_ref, atol=0, rtol=0)


def test_rotary_embedding_qwen3_0_6b_shapes():
    """Qwen3-0.6B 使用 head_dim=128，因此 inv_freq 长度是 64。"""
    rope = RotaryEmbedding(head_dim=128, rope_theta=1_000_000.0)
    x = torch.randn(2, 16, 7, 128)
    position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)

    cos, sin = rope(x, position_ids)

    assert tuple(rope.inv_freq.shape) == (64,)
    assert cos.shape == (2, 7, 128)
    assert sin.shape == (2, 7, 128)
    assert cos.dtype == x.dtype
    assert sin.dtype == x.dtype
