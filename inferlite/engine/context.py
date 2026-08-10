"""ForwardContext + LLMModel Protocol。

cache 和 metadata 不经过模型 __call__ 传递：
- cache 通过 adapter.bind_kv_cache(model) 在初始化时绑定到 Attention 层
- metadata 通过 set_forward_context(metadata) 在每次 forward 前设置
- LLMModel Protocol 定义 engine 层需要的最小模型接口
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import torch

# ── ForwardContext ──


@dataclass
class AttentionMetadata:
    """per-forward 的 attention 元数据。纯 tensor，不含 cache 对象引用。

    对齐 vLLM V1 的 FlashAttentionMetadata。
    cache 对象通过 bind_kv_cache 绑定在 Attention 层上，不经过 metadata 传递。
    """

    num_seqs: int
    seq_lens: torch.Tensor  # [num_seqs] 每个请求的序列长度
    slot_mapping: torch.Tensor | None = None  # [num_tokens] M3 batched
    block_table: torch.Tensor | None = None  # [num_seqs, max_blocks] M4 paged


@dataclass
class ForwardContext:
    """全局前向上下文。对齐 vLLM V1 的 ForwardContext。"""

    attn_metadata: AttentionMetadata


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    assert _forward_context is not None, "ForwardContext not set"
    return _forward_context


def has_forward_context() -> bool:
    """检查 ForwardContext 是否已设置（用于 DecoderLayer 判断新旧路径）。"""
    return _forward_context is not None


@contextmanager
def set_forward_context(attn_metadata: AttentionMetadata):
    global _forward_context
    _forward_context = ForwardContext(attn_metadata)
    try:
        yield
    finally:
        _forward_context = None


# ── LLMModel Protocol ──


class LLMModel(Protocol):
    """最小 LLM 推理协议：input_ids -> logits。

    对齐 vLLM V1 的 model runner 接口（简化版）。

    调用示例：

        # M1: 无 cache，每步 full forward
        logits = model(input_ids)

        # M2/M3/M4: cache 已绑定，metadata 已设置
        with set_forward_context(metadata):
            logits = model(input_ids, positions=positions)

        # 只取最后 N 个位置的 logits（decode 优化）
        logits = model(input_ids, logits_to_keep=1)
    """

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        """返回 logits。

        Args:
            input_ids: [B, T] 形状的 token ids。
            positions: [B, T]，绝对位置。None 时模型内部自动生成 0..T-1。
                decode 阶段必须传绝对位置（如 [[cur_len]]），否则 RoPE 每步都在位置 0。
            logits_to_keep: 若为非 None，只返回最后 logits_to_keep 个位置的 logits。

        Returns:
            logits: [B, T, V] 或 [B, logits_to_keep, V]。
        """
        ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """从 hidden states 计算 logits（lm_head 投影）。

        对齐 vLLM V1 的 logits 与 forward 分离模式。
        当 forward 返回 hidden_states 时使用；当前 forward 直接返回 logits，
        此方法供未来 A5 阶段使用。
        """
        ...
