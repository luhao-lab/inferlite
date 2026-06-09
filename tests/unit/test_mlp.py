"""
Unit tests for inferlite.model.layers.SwiGLUMLP.

T2 目标：手写 Qwen3 的 MLP 子层，并与 transformers.Qwen3MLP 数值对齐。

运行：
  uv run pytest tests/unit/test_mlp.py -q
"""

import pytest
import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from inferlite.model.layers import SwiGLUMLP


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_swiglu_mlp_vs_qwen3_mlp(dtype):
    """与 transformers.Qwen3MLP 在同输入、同权重下数值对齐。

    用小尺寸 config 测逻辑，不用 1024/3072 大尺寸，避免单测太慢。
    """
    # 固定随机种子：确保 ref/mine 看到同一份输入，差异只能来自实现。
    torch.manual_seed(0)
    # 构造一个“迷你 Qwen3 MLP”配置。
    # 这里只测 MLP 公式和权重命名，不需要真实 0.6B 的大矩阵。
    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        hidden_act="silu",
    )
    # transformers 官方 Qwen3MLP 是 T2 的 ground truth。
    # .eval() 固定推理态；虽然 MLP 没 dropout，但数值对齐测试统一这么写。
    ref = Qwen3MLP(cfg).to(dtype).eval()
    # inferlite 手写实现，超参必须和 ref 完全一致。
    mine = (
        SwiGLUMLP(
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
        )
        .to(dtype)
        .eval()
    )

    # 同步三组权重：gate_proj / up_proj / down_proj。
    # 这一步同时验证我们的参数名与 transformers 对齐，否则 load_state_dict 会报 missing/unexpected。
    mine.load_state_dict(ref.state_dict())

    # 常规 [B, T, H] 输入，覆盖 LLM 中 MLP 的真实调用形态。
    x = torch.randn(2, 5, cfg.hidden_size, dtype=dtype)
    with torch.no_grad():
        y_ref = ref(x)
        y_mine = mine(x)

    # fp16/bf16 精度较低，容差放宽；fp32 用更严格阈值。
    atol = 1e-5 if dtype == torch.float32 else 5e-3
    assert y_mine.shape == x.shape
    assert torch.allclose(
        y_mine, y_ref, atol=atol, rtol=1e-4
    ), f"max diff = {(y_mine - y_ref).abs().max().item()}"


def test_swiglu_mlp_bias_is_none():
    """Qwen3 MLP 三个 Linear 都是 bias=False。"""
    mlp = SwiGLUMLP(hidden_size=16, intermediate_size=32)
    # nn.Linear 默认 bias=True；T2 实现必须显式关掉。
    assert mlp.gate_proj.bias is None
    assert mlp.up_proj.bias is None
    assert mlp.down_proj.bias is None


@pytest.mark.parametrize("shape", [(16,), (3, 16), (2, 5, 16), (2, 3, 5, 16)])
def test_swiglu_mlp_shape_invariant(shape):
    """输入 [..., H]，输出仍是 [..., H]。"""
    mlp = SwiGLUMLP(hidden_size=16, intermediate_size=32)
    # 覆盖 1D/2D/3D/4D：实现不能写死 batch/seq 维度，只能约定最后一维是 H。
    x = torch.randn(*shape)
    y = mlp(x)
    assert y.shape == x.shape


def test_swiglu_mlp_has_exactly_three_linear_layers():
    """SwiGLU = gate/up/down 三个 Linear。"""
    mlp = SwiGLUMLP(hidden_size=16, intermediate_size=32)
    # 防止误写成普通 FFN（2 个 Linear）或多加不必要投影。
    linear_layers = [module for module in mlp.modules() if isinstance(module, nn.Linear)]
    assert len(linear_layers) == 3


def test_swiglu_mlp_qwen3_0_6b_weight_shapes():
    """Qwen3-0.6B 的 MLP 权重形状固定为 H=1024, I=3072。"""
    mlp = SwiGLUMLP(hidden_size=1024, intermediate_size=3072)
    # PyTorch Linear.weight 形状是 [out_features, in_features]。
    assert tuple(mlp.gate_proj.weight.shape) == (3072, 1024)
    assert tuple(mlp.up_proj.weight.shape) == (3072, 1024)
    assert tuple(mlp.down_proj.weight.shape) == (1024, 3072)
