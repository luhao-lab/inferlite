"""Unit tests for inferlite.model.qwen3.Qwen3Model.

T6 目标：实现 Qwen3 backbone，即 embed_tokens + DecoderLayer 堆叠 + final RMSNorm。
T6 不做 lm_head/logits，不做真实权重加载。

运行：
  uv run pytest tests/unit/test_qwen3_model.py -q
"""

import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model as RefQwen3Model

from inferlite.config import ModelConfig
from inferlite.model.qwen3 import Qwen3Model


def _tiny_model_config(num_hidden_layers: int = 2) -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
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


def _tiny_qwen3_config(num_hidden_layers: int = 2) -> Qwen3Config:
    return Qwen3Config(
        hidden_size=32,
        num_hidden_layers=num_hidden_layers,
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


def _copy_model_weights(mine: Qwen3Model, ref: RefQwen3Model) -> None:
    """复制小尺寸 transformers.Qwen3Model 权重到 inferlite.Qwen3Model。"""
    mine.embed_tokens.weight.data.copy_(ref.embed_tokens.weight.data)
    mine.norm.weight.data.copy_(ref.norm.weight.data)

    for mine_layer, ref_layer in zip(mine.layers, ref.layers, strict=True):
        mine_layer.self_attn.q_proj.weight.data.copy_(ref_layer.self_attn.q_proj.weight.data)
        mine_layer.self_attn.k_proj.weight.data.copy_(ref_layer.self_attn.k_proj.weight.data)
        mine_layer.self_attn.v_proj.weight.data.copy_(ref_layer.self_attn.v_proj.weight.data)
        mine_layer.self_attn.o_proj.weight.data.copy_(ref_layer.self_attn.o_proj.weight.data)
        mine_layer.self_attn.q_norm.weight.data.copy_(ref_layer.self_attn.q_norm.weight.data)
        mine_layer.self_attn.k_norm.weight.data.copy_(ref_layer.self_attn.k_norm.weight.data)

        mine_layer.mlp.gate_proj.weight.data.copy_(ref_layer.mlp.gate_proj.weight.data)
        mine_layer.mlp.up_proj.weight.data.copy_(ref_layer.mlp.up_proj.weight.data)
        mine_layer.mlp.down_proj.weight.data.copy_(ref_layer.mlp.down_proj.weight.data)

        mine_layer.input_layernorm.weight.data.copy_(ref_layer.input_layernorm.weight.data)
        mine_layer.post_attention_layernorm.weight.data.copy_(
            ref_layer.post_attention_layernorm.weight.data
        )


def test_qwen3_model_structure_and_shapes():
    """模型主体包含 embedding、N 层 DecoderLayer 和 final norm。"""
    cfg = _tiny_model_config(num_hidden_layers=2)
    model = Qwen3Model(cfg)

    assert model.vocab_size == cfg.vocab_size
    assert tuple(model.embed_tokens.weight.shape) == (cfg.vocab_size, cfg.hidden_size)
    assert len(model.layers) == cfg.num_hidden_layers
    assert tuple(model.norm.weight.shape) == (cfg.hidden_size,)


def test_qwen3_model_forward_output_shape():
    """input_ids [B, T] 应输出 last_hidden_state [B, T, H]。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config(num_hidden_layers=2)
    model = Qwen3Model(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))

    with torch.no_grad():
        output = model(input_ids)

    assert output.shape == (2, 5, cfg.hidden_size)


def test_qwen3_model_auto_position_ids_matches_explicit_position_ids():
    """不传 position_ids 时，应等价于显式传入 0..T-1。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config(num_hidden_layers=2)
    model = Qwen3Model(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        auto_output = model(input_ids)
        explicit_output = model(input_ids, position_ids=position_ids)

    assert torch.allclose(auto_output, explicit_output, atol=0, rtol=0)


class _AddLayer(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.value


def test_qwen3_model_layers_are_applied_in_order_with_monkeypatch():
    """用假 layer 锁定 ModuleList 的顺序：embedding -> layer0 -> layer1 -> norm。"""
    cfg = _tiny_model_config(num_hidden_layers=2)
    model = Qwen3Model(cfg).eval()
    model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    model.embed_tokens.weight.data.zero_()
    model.layers = nn.ModuleList([_AddLayer(1.0), _AddLayer(2.0)])
    model.norm = nn.Identity()
    input_ids = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        output = model(input_ids)

    assert torch.equal(output, torch.full((1, 3, cfg.hidden_size), 3.0))


def test_qwen3_model_vs_transformers_qwen3_model_fp32():
    """小尺寸 fp32 下，Qwen3Model 应与 transformers.Qwen3Model 对齐。"""
    torch.manual_seed(0)
    cfg = _tiny_model_config(num_hidden_layers=2)
    ref_cfg = _tiny_qwen3_config(num_hidden_layers=2)
    mine = Qwen3Model(cfg).eval()
    ref = RefQwen3Model(ref_cfg).eval()
    _copy_model_weights(mine, ref)

    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)

    with torch.no_grad():
        output_mine = mine(input_ids, position_ids=position_ids)
        output_ref = ref(
            input_ids=input_ids, position_ids=position_ids, use_cache=False
        ).last_hidden_state

    assert output_mine.shape == output_ref.shape
    assert torch.allclose(output_mine, output_ref, atol=1e-5, rtol=1e-5)


def test_qwen3_0_6b_model_constructs():
    """真实 Qwen3-0.6B 配置可构造，关键模块 shape 正确。"""
    cfg = ModelConfig.qwen3_0_6b()
    model = Qwen3Model(cfg)

    assert tuple(model.embed_tokens.weight.shape) == (151936, 1024)
    assert len(model.layers) == 28
    assert tuple(model.norm.weight.shape) == (1024,)
