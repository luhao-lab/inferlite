"""Unit tests for inferlite.model.attention.Qwen3Attention.

T4 目标：手写 Qwen3 GQA Attention，并在小尺寸配置上对齐
transformers.models.qwen3.modeling_qwen3.Qwen3Attention。

运行：
  uv run pytest tests/unit/test_attention.py -q
"""

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention as RefQwen3Attention,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
)
from transformers.models.qwen3.modeling_qwen3 import (
    repeat_kv as ref_repeat_kv,
)

from inferlite.config import ModelConfig
from inferlite.model.attention import Qwen3Attention, repeat_kv
from inferlite.model.layers import RotaryEmbedding


def _tiny_model_config() -> ModelConfig:
    """构造一个小尺寸 GQA 配置，方便 attention 数值对齐测试。"""
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=1,
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


def _tiny_qwen3_config() -> Qwen3Config:
    """与 _tiny_model_config 等价的 transformers Qwen3Config。"""
    return Qwen3Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=32,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        tie_word_embeddings=False,
    )


def _copy_attention_weights(mine: Qwen3Attention, ref: RefQwen3Attention) -> None:
    """把 transformers attention 的权重复制到 inferlite attention。"""
    mine.q_proj.weight.data.copy_(ref.q_proj.weight.data)
    mine.k_proj.weight.data.copy_(ref.k_proj.weight.data)
    mine.v_proj.weight.data.copy_(ref.v_proj.weight.data)
    mine.o_proj.weight.data.copy_(ref.o_proj.weight.data)
    mine.q_norm.weight.data.copy_(ref.q_norm.weight.data)
    mine.k_norm.weight.data.copy_(ref.k_norm.weight.data)


def test_repeat_kv_vs_transformers():
    """repeat_kv 的 shape 与内容必须和 transformers 完全一致。"""
    hidden_states = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)

    y_ref = ref_repeat_kv(hidden_states, n_rep=3)
    y_mine = repeat_kv(hidden_states, n_rep=3)

    assert y_mine.shape == (2, 6, 3, 4)
    assert torch.equal(y_mine, y_ref)


def test_repeat_kv_noop_when_n_rep_is_one():
    """n_rep=1 时直接返回原张量，避免无意义 reshape。"""
    hidden_states = torch.randn(2, 4, 3, 8)

    y = repeat_kv(hidden_states, n_rep=1)

    assert y is hidden_states


def test_qwen3_0_6b_projection_shapes():
    """Qwen3-0.6B 的 head_dim=128，不能误用 hidden_size / num_heads。"""
    cfg = ModelConfig.qwen3_0_6b()
    attn = Qwen3Attention(cfg)

    assert attn.head_dim == 128
    assert attn.attn.num_key_value_groups == 2
    assert tuple(attn.q_proj.weight.shape) == (16 * 128, 1024)
    assert tuple(attn.k_proj.weight.shape) == (8 * 128, 1024)
    assert tuple(attn.v_proj.weight.shape) == (8 * 128, 1024)
    assert tuple(attn.o_proj.weight.shape) == (1024, 16 * 128)


def test_gqa_attention_output_shape():
    """Qwen3Attention 输入/输出都保持 residual stream 的 [B, T, H] 形状。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config()
    attn = Qwen3Attention(cfg).eval()
    hidden_states = torch.randn(2, 5, cfg.hidden_size)
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        output = attn(position_ids, hidden_states)

    assert output.shape == hidden_states.shape


def test_gqa_attention_vs_transformers_qwen3_attention_fp32():
    """小尺寸 fp32 下，Qwen3Attention 应与 transformers.Qwen3Attention 对齐。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config()
    ref_cfg = _tiny_qwen3_config()
    mine = Qwen3Attention(cfg).eval()
    ref = RefQwen3Attention(ref_cfg, layer_idx=0).eval()
    _copy_attention_weights(mine, ref)

    batch_size = 2
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, cfg.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # transformers.Qwen3Attention.forward 接收预先算好的 (cos, sin)。
    ref_rope = Qwen3RotaryEmbedding(ref_cfg).eval()
    with torch.no_grad():
        cos_ref, sin_ref = ref_rope(hidden_states, position_ids)
        causal_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
            torch.finfo(hidden_states.dtype).min,
        )
        output_ref, _ = ref(
            hidden_states,
            position_embeddings=(cos_ref, sin_ref),
            attention_mask=causal_mask,
        )
        output_mine = mine(position_ids, hidden_states)

    assert output_mine.shape == output_ref.shape
    assert torch.allclose(output_mine, output_ref, atol=1e-5, rtol=1e-5)


def test_gqa_attention_causal_mask_blocks_future_tokens():
    """修改未来 token 不应影响过去 token 的 attention 输出。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config()
    attn = Qwen3Attention(cfg).eval()
    seq_len = 5
    position_ids = torch.arange(seq_len).unsqueeze(0)
    hidden_states = torch.randn(1, seq_len, cfg.hidden_size)
    changed_future = hidden_states.clone()
    changed_future[:, 3:, :] = torch.randn_like(changed_future[:, 3:, :]) * 10

    with torch.no_grad():
        output = attn(position_ids, hidden_states)
        output_changed = attn(position_ids, changed_future)

    # 第 0~2 个位置只能看见自己和过去，因此不应被第 3~4 个未来 token 影响。
    assert torch.allclose(output[:, :3, :], output_changed[:, :3, :], atol=1e-5, rtol=1e-5)
    # 第 3~4 个位置自身输入变了，输出通常会随之变化。
    assert not torch.allclose(output[:, 3:, :], output_changed[:, 3:, :])


def test_rotary_embedding_used_by_attention_matches_transformers():
    """Attention 内部 RoPE 生成器仍沿用 T3 的 transformers 对齐语义。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config()
    ref_cfg = _tiny_qwen3_config()
    mine_rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta).eval()
    ref_rope = Qwen3RotaryEmbedding(ref_cfg).eval()
    q = torch.randn(2, cfg.num_attention_heads, 5, cfg.head_dim)
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        cos_mine, sin_mine = mine_rope(q, position_ids)
        cos_ref, sin_ref = ref_rope(q, position_ids)

    assert torch.allclose(cos_mine, cos_ref, atol=1e-6, rtol=1e-5)
    assert torch.allclose(sin_mine, sin_ref, atol=1e-6, rtol=1e-5)
