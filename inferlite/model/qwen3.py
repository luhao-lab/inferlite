"""Qwen3 decoder layer 的最小数值对齐实现。

T5 的目标是把前面已经手写并测试过的叶子模块串起来：

    RMSNorm -> GQAAttention -> residual add -> RMSNorm -> SwiGLUMLP -> residual add

Qwen3 采用 pre-norm 结构：先归一化，再进入 attention/MLP，最后与 residual 相加。

pre-norm 与 post-norm 对比：

1. Pre-norm（现代 decoder-only LLM 主流）

       x -> norm -> sublayer -> + residual

   写成代码就是：

       residual = x
       x = norm(x)
       x = sublayer(x)
       x = residual + x

   特点：
   - residual 主干更像一条“干净高速通道”，梯度可以更稳定地穿过很多层；
   - 深层 Transformer 更容易训练，因此 GPT-2 之后的大量 decoder-only LLM 都采用 pre-norm；
   - LLaMA / Qwen / Mistral / Gemma 等现代开源主流大模型基本都是 pre-norm 变体，通常配 RMSNorm。

2. Post-norm（原始 Transformer 论文结构）

       x -> sublayer -> + residual -> norm

   写成代码就是：

       residual = x
       x = sublayer(x)
       x = residual + x
       x = norm(x)

   特点：
   - 原始 Transformer 使用 post-norm；
   - 浅层或 encoder 场景仍可见；
   - 很深的 decoder-only LLM 中通常更难稳定训练，往往需要更谨慎的初始化、学习率 warmup 或其他稳定化技巧。

结论：最新主流 decoder-only 大模型以 pre-norm 为主；本项目为了对齐 Qwen3，也实现 pre-norm。
"""

from typing import override

import torch
import torch.nn as nn

from inferlite.config import ModelConfig
from inferlite.model.attention import GQAAttention
from inferlite.model.layers import RMSNorm, SwiGLUMLP


class DecoderLayer(nn.Module):
    """Qwen3 单个 decoder block: self-attention + MLP。

    一个 decoder layer 有两段子层：
    1. self-attention 子层：负责 token 间信息交互。
    2. MLP 子层：负责每个 token 位置上的非线性特征变换。

    两段都采用相同的 pre-norm + residual 模式。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # hidden_size 是 residual stream 的宽度，DecoderLayer 输入输出都保持这个维度。
        self.hidden_size: int = config.hidden_size

        # self_attn 对应 transformers.Qwen3DecoderLayer.self_attn。
        # 命名对齐很重要：T7 加载 HF 权重时会依赖 state_dict key 的模块路径。
        self.self_attn: GQAAttention = GQAAttention(config)

        # MLP 是 attention 后的前馈网络，对每个 token 独立作用，不做 token 间通信。
        self.mlp: SwiGLUMLP = SwiGLUMLP(config.hidden_size, config.intermediate_size)

        # 两个 RMSNorm 都归一化 hidden_size 维，而不是 attention 的 head_dim。
        # input_layernorm: attention 前的 pre-norm。
        self.input_layernorm: RMSNorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # post_attention_layernorm: MLP 前的 pre-norm。
        # 名字里的 post_attention 表示“attention 之后”，不是 post-norm 结构。
        self.post_attention_layernorm: RMSNorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """执行一个 Qwen3 decoder layer。

        Args:
            hidden_states: [B, T, hidden_size]
                当前层输入的 residual stream。
            position_ids: [B, T]
                传给 attention 内部 RoPE，用于给 q/k 注入位置信息。

        Returns:
            [B, T, hidden_size]
        """
        # 1. Attention 子层：pre-norm -> self-attn -> residual add。
        #
        # 为什么先保存 residual？
        # residual 是“未经本子层变换”的主干信号，最后要和 attention 的增量相加。
        # pre-norm 中，norm 只作用在送进子层的分支上，不修改 residual 主干。
        residual = hidden_states
        # input_layernorm: [B, T, H] -> [B, T, H]
        # 这里归一化每个 token 的 hidden_size 维，让 attention 输入尺度更稳定。
        hidden_states = self.input_layernorm(hidden_states)
        # self_attn 内部会执行：q/k/v projection -> q/k norm -> RoPE -> GQA attention -> o_proj。
        # 输出仍是 [B, T, H]，表示 attention 子层计算出的“增量”。
        hidden_states = self.self_attn(hidden_states, position_ids)
        # residual add：把 attention 增量加回主干。
        hidden_states = residual + hidden_states

        # 2. MLP 子层：重新记录 residual，再 pre-norm -> MLP -> residual add。
        #
        # 注意这里必须重新设置 residual。
        # 第二段 residual 的起点是“已经经过 attention 子层后的 hidden_states”，
        # 不能复用 attention 前的 residual，否则会跳过 attention 子层的贡献。
        residual = hidden_states
        # post_attention_layernorm 虽然名字里有 post_attention，但结构上仍然是 MLP 前的 pre-norm。
        hidden_states = self.post_attention_layernorm(hidden_states)
        # MLP 对每个 token 独立做 SwiGLU 变换，不改变 [B, T, H] 形状。
        hidden_states = self.mlp(hidden_states)
        # 第二次 residual add：把 MLP 增量加回主干。
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):
    """Qwen3 backbone：embedding + N 个 DecoderLayer + final RMSNorm。

    从接口语义上看，它对应 transformers 里的 `Qwen3Model`，也就是“裸模型主干”：

        input_ids -> token embeddings -> decoder layers -> final norm -> last_hidden_state

    注意它不是 `Qwen3ForCausalLM`：
    - 不包含 `lm_head`
    - 不输出 vocab logits
    - 不计算 loss
    - 不做采样 / 生成循环

    T6 只返回 last_hidden_state；T7 负责真实权重加载，T8 才会接 lm_head 做 logits 对齐。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # 保存完整 config，后续 T7/T8 可能需要读取层数、词表大小、tie_word_embeddings 等信息。
        self.config: ModelConfig = config
        # Qwen3Config 的 pad_token_id 在某些场景可能为 None；T6 暂不处理 padding 语义。
        # 这里保留 padding_idx 字段主要是为了结构上贴近 transformers.Qwen3Model。
        self.padding_idx: int = 0
        self.vocab_size: int = config.vocab_size

        # token embedding：把离散 token id 映射成 residual stream 向量。
        # input_ids: [B, T] -> hidden_states: [B, T, H]
        # 这里还没有 position embedding，因为 Qwen3 使用 RoPE；位置信息会在 attention 的 q/k 上注入。
        self.embed_tokens: nn.Embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
        )

        # decoder layer 堆叠：每层都是 T5 实现的 DecoderLayer。
        # ModuleList 会把子模块注册进 state_dict，T7 加载权重时会形成类似：
        #   layers.0.self_attn.q_proj.weight
        #   layers.1.mlp.gate_proj.weight
        # 的层级路径。
        self.layers: nn.ModuleList = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # final norm：所有 decoder layers 之后的最后一次 RMSNorm。
        # 它仍然作用在 hidden_size 维上，输出 last_hidden_state，供后续 lm_head 计算 logits。
        self.norm: RMSNorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    @override
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行 Qwen3 主干前向，返回最后一层 hidden states。

        Args:
            input_ids: [B, T]
                token ids，来自 tokenizer encode 后的整数序列。
            position_ids: [B, T]。如果不传，则按 0..T-1 自动生成。
                T6 只处理无 KV cache 的 full forward，因此每个 batch 都从 0 开始。

        Returns:
            last_hidden_state: [B, T, hidden_size]
        """
        batch_size, seq_len = input_ids.shape
        if position_ids is None:
            # 默认 position ids 是每个序列从 0 到 T-1。
            # arange 必须放在 input_ids.device 上，避免 CPU/MPS/CUDA 混用。
            # expand 到 [B, T] 后，每个 batch 共享同一套位置编号。
            position_ids = (
                torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            )

        # Step 1: token id -> token embedding。
        # hidden_states 是后续所有 decoder layers 传递的 residual stream。
        hidden_states = self.embed_tokens(input_ids)

        # Step 2: 逐层通过 decoder blocks。
        # 每层都接收同一份 position_ids，用于内部 attention 的 RoPE。
        # transformers 当前实现会在 model 层预先算 position_embeddings 再传给每层；
        # inferlite 当前实现让每层 attention 自己根据 position_ids 计算 cos/sin。
        # 两者接口不同，但数值目标等价。
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids)

        # Step 3: final RMSNorm。
        # 这是 Qwen3Model 输出前的最后归一化；lm_head/logits 会在后续 T8 接在它后面。
        hidden_states = self.norm(hidden_states)
        return hidden_states
