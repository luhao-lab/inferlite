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
import torch.nn.functional as F


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


class SwiGLUMLP(nn.Module):
    """Qwen3 的 SwiGLU MLP 子层。

    结构：
        x ── gate_proj ── SiLU ─┐
                                ├─ element-wise multiply ─ down_proj ─ y
        x ── up_proj   ─────────┘

    公式：
        y = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        hidden_size: residual stream 维度 H，Qwen3-0.6B 为 1024。
        intermediate_size: MLP 中间维度 I，Qwen3-0.6B 为 3072。

    Shape:
        Input:  [..., hidden_size]       常见 [B, T, H]
        Output: [..., hidden_size]       MLP 不改变 residual stream 维度
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # gate 路：产生“软门控”信号，后续只对这一路做 SiLU。
        # 形状：[..., H] -> [..., I]
        # Qwen3 的 MLP Linear 都是 bias=False，必须显式写出来，不能用 nn.Linear 默认值。
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # up 路：产生被 gate 调制的“内容”信号。
        # 形状同 gate 路：[..., H] -> [..., I]
        # 注意不要和 gate_proj 交换命名；T7 加载 HF 权重时 key 会严格区分 gate/up。
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # down 路：把 SwiGLU 中间维度 I 压回 residual stream 维度 H。
        # 形状：[..., I] -> [..., H]
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: 两条独立线性投影。
        # gate/up 都是 [..., I]，但语义不同：gate 控制开关，up 提供内容。
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Step 2: 只对 gate 路做 SiLU，再逐元素乘 up 路。
        # 正确：silu(gate) * up
        # 错误：silu(gate * up) —— 这会改变 SwiGLU 的定义，无法对齐 transformers.Qwen3MLP。
        hidden = F.silu(gate) * up

        # Step 3: 回到 hidden_size，供 residual add 使用。
        return self.down_proj(hidden)
