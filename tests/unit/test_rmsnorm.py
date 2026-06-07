"""
Unit tests for inferlite.model.layers.RMSNorm

测试矩阵：
  - 测试 1（参数化）: 3 个 shape × 3 个 dtype = 9 个 case，与 transformers Qwen3RMSNorm 数值对齐
  - 测试 2: shape 不变性（任意前缀维度都能工作）
  - 测试 3: weight 是 nn.Parameter、可学、初值全 1
  - 测试 4: eps 默认 1e-6（Qwen3 用 1e-6，区别于 LLaMA 的 1e-5）

为什么不只测一种 case？
  RMSNorm 看似简单（4 行内核），但典型 bug 都藏在边界里：
    - 升 fp32 算方差漏写 → fp16/bf16 case 数值爆炸（测试 1 的 dtype 维度兜底）
    - mean(dim=-1, keepdim=True) 漏 keepdim → shape=(64,) 一维输入直接报错（测试 2 的最短 shape）
    - weight 没注册成 Parameter → optimizer.step() 无效（测试 3）
    - eps 抄成 1e-5 → logits 在小幅度场景差 1e-3 量级，整模累计就崩（测试 4）

运行：
  uv run pytest tests/unit/test_rmsnorm.py -v
"""

import pytest
import torch

# 引用 transformers 官方实现作为 "ground truth"
# 我们的实现必须与 Qwen3RMSNorm 数值对齐才算合格
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from inferlite.model.layers import RMSNorm


# ─────────────────────────────────────────────────────────────────────────────
# 测试 1: 跨 shape × dtype 与 transformers 官方实现数值对齐
# ─────────────────────────────────────────────────────────────────────────────
# pytest.mark.parametrize 会把下面的 dtype × shape 笛卡尔积展开成 9 个独立 case
# 任何一个 case 红 → pytest 单独标 FAIL，方便定位是哪个组合崩了
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 5, 1024), (1, 1, 1024), (8, 128, 1024)])
def test_rmsnorm_vs_qwen3(dtype, shape):
    """与 transformers.Qwen3RMSNorm 在不同 (shape, dtype) 下数值一致。

    shape 选择理由：
      (2, 5, 1024)   常规小 batch / 短序列（最典型）
      (1, 1, 1024)   退化 batch=seq=1（边界，避免 squeeze 类 bug）
      (8, 128, 1024) 大一点的张量（暴露 reduce 时数值累积误差）

    dtype 选择理由：
      fp32  推理精度上限，atol=1e-5 是数值的"地板"
      fp16  Mac MPS / 老 GPU 常用，尾数 10 bit，atol=1e-3
      bf16  现代加速器主流（H100/B200/MPS），范围大但尾数 7 bit，atol=1e-3
    """
    H = shape[-1]  # hidden_size 总是最后一维（RMSNorm 沿这一维归一化）

    # 固定随机种子，让 ours / ref 看到完全一样的输入
    # 这是数值对齐测试的基本纪律：差异必须来自实现，不能来自输入噪声
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype)

    # 两份实现都按相同超参构造（H, eps），并搬到目标 dtype
    # 注意：.to(dtype) 会把 weight 也转过去，模拟真实推理中权重就是 fp16/bf16 的场景
    ours = RMSNorm(H, eps=1e-6).to(dtype)
    ref: Qwen3RMSNorm = Qwen3RMSNorm(H, eps=1e-6).to(dtype)

    # 同步权重 —— 虽然两边默认都 init 成全 1（恒等变换），
    # 但显式 copy 一次能防御未来某天 init 策略改了导致测试"莫名"挂掉
    # .data.copy_() 绕过 autograd 直接覆写底层 tensor，标准做法
    ref.weight.data.copy_(ours.weight.data)

    # 前向：同一个 x 喂给两份实现
    out_ours = ours(x)
    out_ref = ref(x)

    # 误差预算：
    #   fp32   原生 ~1e-7，留 100× 余量到 1e-5
    #   fp16/bf16  原生 ~1e-3，再松也没意义（已是 dtype 精度极限）
    atol = 1e-5 if dtype == torch.float32 else 1e-3

    # 断言 1: 形状必须严格等于输入（RMSNorm 不改变 shape，只 normalize 最后一维）
    assert out_ours.shape == x.shape

    # 断言 2: 数值在容差范围内对齐
    # torch.allclose 判定: |a - b| <= atol + rtol * |b|（逐元素）
    # 失败时打印 max diff，方便定位是"差一点点"还是"完全错"
    assert torch.allclose(
        out_ours, out_ref, atol=atol, rtol=1e-4
    ), f"max diff = {(out_ours - out_ref).abs().max().item()}"


# ─────────────────────────────────────────────────────────────────────────────
# 测试 2: shape invariant —— RMSNorm 必须对任意前缀维度都成立
# ─────────────────────────────────────────────────────────────────────────────
def test_rmsnorm_shape_invariant():
    """输入输出 shape 完全一致；前缀维度可以是 0/1/2/3 个。

    为什么单测这条？
      RMSNorm 在 LLM 里会被嵌入到 [B, T, H] 三维张量里使用，
      但在调试 / unit test 时常常会用 [H] 或 [T, H] 这种更小的形状。
      若实现里写死了 dim=2，遇到 [H] 一维输入会直接 IndexError。
      正确做法：用 dim=-1 + keepdim=True 让最后一维永远是被 reduce 的那一维。
    """
    layer = RMSNorm(64)
    # 4 种典型形状，覆盖 1D / 2D / 3D / 4D
    for shape in [(64,), (3, 64), (2, 7, 64), (2, 3, 5, 64)]:
        x = torch.randn(*shape)
        y = layer(x)
        assert y.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# 测试 3: weight 是可学参数且初始化为全 1
# ─────────────────────────────────────────────────────────────────────────────
def test_rmsnorm_weight_learnable():
    """weight 必须是 nn.Parameter，requires_grad=True，初始值 = ones。

    为什么要测？
      常见 bug:
        self.weight = torch.ones(H)          # 普通 tensor，optimizer 看不到 ❌
        self.weight = nn.Parameter(...)      # 正确 ✅
      训练时这种 bug 不会立即报错，但 loss 不会下降；推理时也不会报错，
      但 from_pretrained 加载权重时 state_dict key 缺失 → 默默漏权重。
      单测在这里把"参数登记"这件事钉死。
    """
    layer = RMSNorm(16)
    # 断言 1: 类型是 nn.Parameter（不是普通 Tensor）
    # nn.Parameter 是 Tensor 的子类，会被 nn.Module 自动登记到 .parameters()
    assert isinstance(layer.weight, torch.nn.Parameter)
    # 断言 2: 默认参与梯度计算（推理时会被 no_grad 关掉，但参数本身得是可学的）
    assert layer.weight.requires_grad
    # 断言 3: 初值 = 1（恒等变换的中立起点；训练初期 RMSNorm 退化成纯归一化）
    assert torch.allclose(layer.weight, torch.ones(16))


# ─────────────────────────────────────────────────────────────────────────────
# 测试 4: 默认 eps = 1e-6（Qwen3 专属）
# ─────────────────────────────────────────────────────────────────────────────
def test_rmsnorm_eps_default_1e_6():
    """Qwen3 系列用 eps=1e-6，区别于 LLaMA 系列的 eps=1e-5。

    为什么这个细节很重要？
      eps 是 rsqrt(var + eps) 里那个数值稳定项。
      LLaMA / Qwen2 用 1e-5，Qwen3 改成 1e-6 → 归一化更"敏感"。
      如果你照着 LLaMA 教程抄 eps=1e-5，整个 LLM 的 logits 会在小幅度 hidden 上偏差 1e-3 量级，
      28 层叠加后 token 选择都可能不一样（贪心解码出"另一句话"）。
    """
    layer = RMSNorm(8)
    # 注：这里读的属性名要与你的实现对应；
    # 当前实现里属性叫 self.eps（与 transformers / LLaMA / Mistral 社区约定一致）
    assert layer.eps == 1e-6
