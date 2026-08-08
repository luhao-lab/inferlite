"""Unit tests for inferlite.model.qwen3.DecoderLayer.

T5 目标：把 T1/T2/T4 的模块串成 Qwen3 decoder layer，并与
transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer 小尺寸 fp32 对齐。

运行：
  uv run pytest tests/unit/test_decoder_layer.py -q
"""

import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
)

from inferlite.config import ModelConfig
from inferlite.model.qwen3 import DecoderLayer


def _tiny_model_config() -> ModelConfig:
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


def _copy_decoder_layer_weights(mine: DecoderLayer, ref: Qwen3DecoderLayer) -> None:
    """把 transformers decoder layer 的同名权重复制到 inferlite layer。"""
    mine.self_attn.q_proj.weight.data.copy_(ref.self_attn.q_proj.weight.data)
    mine.self_attn.k_proj.weight.data.copy_(ref.self_attn.k_proj.weight.data)
    mine.self_attn.v_proj.weight.data.copy_(ref.self_attn.v_proj.weight.data)
    mine.self_attn.o_proj.weight.data.copy_(ref.self_attn.o_proj.weight.data)
    mine.self_attn.q_norm.weight.data.copy_(ref.self_attn.q_norm.weight.data)
    mine.self_attn.k_norm.weight.data.copy_(ref.self_attn.k_norm.weight.data)

    mine.mlp.gate_proj.weight.data.copy_(ref.mlp.gate_proj.weight.data)
    mine.mlp.up_proj.weight.data.copy_(ref.mlp.up_proj.weight.data)
    mine.mlp.down_proj.weight.data.copy_(ref.mlp.down_proj.weight.data)

    mine.input_layernorm.weight.data.copy_(ref.input_layernorm.weight.data)
    mine.post_attention_layernorm.weight.data.copy_(ref.post_attention_layernorm.weight.data)


def test_decoder_layer_structure_matches_qwen3_names():
    """子模块命名要对齐 transformers，方便 T7 加载权重。"""
    layer = DecoderLayer(_tiny_model_config())

    assert hasattr(layer, "self_attn")
    assert hasattr(layer, "mlp")
    assert hasattr(layer, "input_layernorm")
    assert hasattr(layer, "post_attention_layernorm")


def test_decoder_layer_output_shape():
    """DecoderLayer 不改变 residual stream 形状。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config()
    layer = DecoderLayer(cfg).eval()
    hidden_states = torch.randn(2, 5, cfg.hidden_size)
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        output = layer(hidden_states, position_ids)

    assert output.shape == hidden_states.shape


class _AddOne(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _AddTwoAttention(nn.Module):
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states + 2


def test_decoder_layer_residual_order_with_monkeypatch():
    """用可手算的假模块锁定 pre-norm + 两段 residual 顺序。"""
    cfg = _tiny_model_config()
    layer = DecoderLayer(cfg).eval()
    layer.input_layernorm = _AddOne()
    layer.self_attn = _AddTwoAttention()
    layer.post_attention_layernorm = _AddOne()
    layer.mlp = _AddOne()

    hidden_states = torch.zeros(1, 2, cfg.hidden_size)
    position_ids = torch.arange(2).unsqueeze(0)

    with torch.no_grad():
        output = layer(hidden_states, position_ids)

    # attention 段：0 -> norm(+1) -> attn(+2) = 3；residual add 后 = 3。
    # mlp 段：3 -> norm(+1) -> mlp(+1) = 5；residual add 后 = 8。
    assert torch.equal(output, torch.full_like(hidden_states, 8.0))


def test_decoder_layer_vs_transformers_qwen3_decoder_layer_fp32():
    """小尺寸 fp32 下，DecoderLayer 应与 transformers.Qwen3DecoderLayer 对齐。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config()
    ref_cfg = _tiny_qwen3_config()
    mine = DecoderLayer(cfg).eval()
    ref = Qwen3DecoderLayer(ref_cfg, layer_idx=0).eval()
    _copy_decoder_layer_weights(mine, ref)

    batch_size = 2
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, cfg.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    ref_rope = Qwen3RotaryEmbedding(ref_cfg).eval()
    with torch.no_grad():
        cos_ref, sin_ref = ref_rope(hidden_states, position_ids)
        causal_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
            torch.finfo(hidden_states.dtype).min,
        )
        output_ref = ref(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            position_embeddings=(cos_ref, sin_ref),
        )
        output_mine = mine(hidden_states, position_ids)

    assert output_mine.shape == output_ref.shape
    assert torch.allclose(output_mine, output_ref, atol=1e-5, rtol=1e-5)


def test_decoder_layer_qwen3_0_6b_constructs():
    """真实 Qwen3-0.6B 配置可构造，关键子模块 shape 正确。"""
    cfg = ModelConfig.qwen3_0_6b()
    layer = DecoderLayer(cfg)

    assert layer.hidden_size == 1024
    assert tuple(layer.input_layernorm.weight.shape) == (1024,)
    assert tuple(layer.post_attention_layernorm.weight.shape) == (1024,)
    assert tuple(layer.self_attn.q_proj.weight.shape) == (16 * 128, 1024)
    assert tuple(layer.mlp.gate_proj.weight.shape) == (3072, 1024)
