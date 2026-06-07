import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no mean-centering, no bias).
    Equivalent to transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.

    LayerNorm(x) = (x - μ) / σ * γ + β，两个统计量、两个可学参数。
    RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ，一个统计量、一个可学参数。

    Args:
        hidden_size (int): The size of the input tensor.
        eps (float): A small value added to the denominator for numerical stability. Default is 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype  # 记原 dtype = x.dtype
        x = x.to(torch.float32)  #  # 必须升 fp32 算方差，否则 fp16/bf16 数值爆炸
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_eps)  # rsqrt 比 1/sqrt 快且数值更稳
        return (self.weight * x).to(input_dtype)
