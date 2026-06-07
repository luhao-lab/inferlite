"""
inferlite.model.layers

M1 阶段手撕的"叶子模块"：纯计算、无状态、无外部依赖。
当前包含：
    - RMSNorm  : Qwen3 / LLaMA 系列归一化层

后续会追加：
    - SwiGLUMLP : Qwen3 MLP
    - RotaryEmbedding : RoPE 位置编码
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm —— Qwen3 / LLaMA / Mistral 等模型用的归一化层。

    与传统 LayerNorm 的对比：
        LayerNorm(x) = (x - μ) / σ * γ + β        # 两个统计量(μ, σ) + 两个参数(γ, β)
        RMSNorm(x)   = x / sqrt(mean(x²) + ε) * γ # 一个统计量(RMS)   + 一个参数(γ)

    核心思想：砍掉均值中心化（μ）和偏置（β），只做"按尺度缩放"。
    实测精度几乎不掉，但 reduce 操作少一半，是现代 LLM 的标配。

    与 transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm 数值完全对齐。

    Args:
        hidden_size: 输入张量最后一维的大小 (Qwen3-0.6B = 1024)
        eps: 加在 sqrt 里的数值稳定项，防止方差极小时 rsqrt 爆炸。
             Qwen3 用 1e-6（注意：LLaMA / Qwen2 用 1e-5，不要照抄）

    Shape:
        Input:  [..., hidden_size]   任意前缀维度，常见 [B, T, H]
        Output: 同 input shape       只沿最后一维归一化
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        # weight (γ) 是唯一的可学参数，shape=[hidden_size]
        # 初始化为全 1 —— 相当于"恒等变换"的起点，训练初期 RMSNorm 退化成纯归一化
        # 必须包成 nn.Parameter，否则:
        #   - optimizer 看不到（训练时不更新）
        #   - state_dict 不会保存它（from_pretrained 漏权重）
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # eps 不是可学参数，普通 float 字段即可
        # 命名跟随社区约定（transformers / LLaMA / Mistral 都叫 .eps 或 .variance_epsilon）
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: 记住原 dtype（fp16/bf16 来的就回 fp16/bf16）
        input_dtype = x.dtype

        # Step 2: 升 fp32 算方差
        # 为什么必须升？fp16 的最大值约 65504，hidden_size=1024 时
        # mean(x²) 容易溢出/丢精度。bf16 范围够但尾数只有 7 位，平方后精度损失更大。
        # 业界共识：归一化的 reduce 阶段一律 fp32，不可省。
        x = x.to(torch.float32)

        # Step 3: 沿最后一维算 mean(x²) = E[x²]
        # 这是"均方根 RMS"的平方：RMS(x) = sqrt(mean(x²))
        # keepdim=True 让 var 形状变成 [..., 1]，方便后面广播除法
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # Step 4: 归一化
        # 等价于 x / sqrt(var + eps)，但用 rsqrt 走硬件指令，更快更稳
        # eps 必须加在 sqrt 里面：rsqrt(var + eps) 而不是 1 / (sqrt(var) + eps)
        x = x * torch.rsqrt(var + self.eps)

        # Step 5: 乘以可学的 scale 参数 γ，再转回原 dtype
        # 顺序很关键：先乘 weight（fp32 × fp32，精度最好），再 .to(input_dtype)
        # 反过来"先 cast weight 再乘" 会损失精度
        return (self.weight * x).to(input_dtype)
