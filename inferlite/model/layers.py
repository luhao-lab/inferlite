"""
inferlite.model.layers

M1 阶段手撕的"叶子模块"：纯计算、无状态、无外部依赖。
当前包含：
    - RMSNorm  : Qwen3 / LLaMA 系列归一化层

后续会追加：
    - SwiGLUMLP : Qwen3 MLP
    - RotaryEmbedding : RoPE 位置编码

参考: https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/modeling_qwen3.py
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """构造 RoPE 二维旋转公式里的 sin 方向项。

    RoPE 最终旋转角度不是 90 度，而是 position_id * inv_freq。
    这里的 rotate_half 只是把二维向量 [x, y] 变成 [-y, x]，也就是先构造
    “逆时针 90 度方向”的基向量，用来写出任意角度旋转：

        [x', y'] = [x, y] * cosθ + [-y, x] * sinθ

    Qwen3/transformers 采用的是“前半 + 后半”配对：

        x = [x1, x2]  ->  rotate_half(x) = [-x2, x1]

    例如 [1, 2, 3, 4] -> [-3, -4, 1, 2]。
    注意：这里不是 even/odd interleave，也不能写成 [-x1, x1]。
    """
    # x1 / x2 形状都为 [..., head_dim // 2]。
    # Qwen3 的 rotate_half 与 transformers 完全一致，T3 数值对齐依赖这个细节。
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 RoPE 生成的 cos/sin 应用到 attention 的 q/k 上。

    Args:
        q: query，T4 Attention 中预期形状为 [B, num_heads, T, head_dim]。
        k: key，T4 Attention 中预期形状为 [B, num_kv_heads, T, head_dim]。
        cos/sin: RotaryEmbedding 输出，形状为 [B, T, head_dim]。
        unsqueeze_dim: 把 cos/sin 扩一维以便广播。
            - q/k 为 [B, heads, T, D] 时用 1，变成 [B, 1, T, D]
            - q/k 为 [B, T, heads, D] 时用 2，变成 [B, T, 1, D]

    RoPE 只作用于 q/k，不作用于 v；相对位置信息会通过 q·k 点积进入 attention score。

    具体维度配对例子（head_dim=8）：
        q = [q0, q1, q2, q3, q4, q5, q6, q7]
        rotate_half(q) = [-q4, -q5, -q6, -q7, q0, q1, q2, q3]

    因为 emb = concat(freqs, freqs)，所以 cos0==cos4、sin0==sin4。
    最终第 0/4 维会组成一个二维旋转平面：
        q_rot[0] = q0*cos0 - q4*sin0
        q_rot[4] = q4*cos0 + q0*sin0

    因此 Qwen3 RoPE 的配对方式是 i 和 i + head_dim/2 配对；
    head_dim=128 时就是 0↔64, 1↔65, ..., 63↔127。
    """
    # [B, T, D] -> [B, 1, T, D]，对所有 heads 共享同一套 position cos/sin。
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # 复数旋转的实数写法：x * cos + rotate_half(x) * sin。
    # q/k 都要旋转；v 是被加权求和的内容向量，不注入 RoPE。
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """Qwen3 默认 RoPE 的 cos/sin 生成器。

    T3 只实现 default RoPE：给定 position_ids，按 head_dim 和 rope_theta 计算
    每个 token 位置对应的 cos/sin 表。它本身不旋转 q/k；旋转由
    apply_rotary_pos_emb 完成。

    为什么 forward 要传入 x（实际调用时常传 q）？
        x 不参与角度计算；真正决定角度的是 position_ids、inv_freq、rope_theta。
        传 x 只是为了复用它的 device 和 dtype：
        - cos/sin 要和 q/k 在同一设备上，避免 CPU/MPS/CUDA 混用；
        - cos/sin 最终要 cast 回 q/k 的 dtype，避免后续 attention 计算类型不一致。
        因此 T4 里写 self.rotary_emb(q, position_ids) 的含义是：
        “按 position_ids 生成 RoPE 表，并让这张表跟 q 的 dtype/device 对齐”。

    Shape:
        x:            任意含 dtype/device 的参考张量，常见 [B, heads, T, head_dim]
        position_ids: [B, T]
        cos/sin:      [B, T, head_dim]

    与 nano-vllm 的主要区别：
        - inferlite 当前实现每次 forward 按 position_ids 现算 cos/sin，便于对齐 transformers。
        - nano-vllm 在 __init__ 里预先缓存 [max_position, ...] 的 cos/sin，forward 只索引缓存，
          更适合推理性能，但对 T3 学习和逐项对齐不如当前写法直观。
    """

    def __init__(self, head_dim: int, rope_theta: float = 1000000.0) -> None:
        super().__init__()
        # inv_freq 是每一对 2D 旋转平面的“角速度”：维度越靠后，频率越低。
        # arange(0, head_dim, 2) 只取偶数位置，所以长度是 head_dim // 2。
        # Qwen3-0.6B: head_dim=128 -> inv_freq.shape == [64]。
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # inv_freq 不是可学习参数，不能放进 nn.Parameter；但它需要跟随 module.to(device)。
        # register_buffer(..., persistent=False) 表示：注册成模块状态、跟随 device，
        # 但不保存进 state_dict，因为它可由 head_dim/rope_theta 重新计算。
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # basedpyright 不知道 register_buffer 会动态创建 self.inv_freq；这行只为静态类型检查。
        self.inv_freq: torch.Tensor = self.inv_freq

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据 position_ids 生成与 x dtype/device 对齐的 cos/sin。

        x 只作为 dtype/device 参考张量；角度本身不依赖 x 的数值。
        """
        # self.inv_freq: [D/2]
        # 扩成 [1, D/2, 1]，后面与 [B, 1, T] 的 position_ids 做 batched matmul。
        inv_freq = self.inv_freq[None, :, None].float().to(x.device)
        # position_ids: [B, T] -> [B, 1, T]
        # 用 float 参与频率乘法；position 本身仍然是整数语义。
        position_ids = position_ids[:, None, :].float()

        # 强制三角函数阶段使用 fp32，避免 fp16/bf16 的精度损失。
        # 这和 transformers.Qwen3RotaryEmbedding 的实现意图一致。
        with torch.autocast(device_type=x.device.type, enabled=False):
            # [1, D/2, 1] @ [B, 1, T] -> [B, D/2, T]
            # transpose 后变成 [B, T, D/2]，方便最后一维拼回 head_dim。
            freqs = (inv_freq @ position_ids).transpose(1, 2)
            # transformers/Qwen3 使用 concat(freqs, freqs)，得到 [B, T, D]。
            # 后续 rotate_half 会把前半/后半配对成二维旋转平面。
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # 返回 dtype 与输入 x 一致；attention 计算里 q/k 通常是 fp16/bf16。
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
