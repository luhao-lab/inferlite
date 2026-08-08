from contextlib import contextmanager
from dataclasses import dataclass

import torch


@dataclass
class AttentionMetadata:
    """per-forward 的 attention 元数据。纯 tensor，不含 cache 对象引用。

    对齐 vLLM V1 的 FlashAttentionMetadata。
    cache 对象通过 bind_kv_cache 绑定在 Attention 层上，不经过 metadata 传递。
    """

    num_seqs: int  # batch 中的请求数
    seq_lens: torch.Tensor  # [num_seqs] 每个请求的序列长度
    slot_mapping: torch.Tensor | None = None  # [num_tokens] M3 batched 路径的 slot 映射
    block_table: torch.Tensor | None = None  # [num_seqs, max_blocks] M4 paged 路径的 block 表


@dataclass
class ForwardContext:
    """全局前向上下文。对齐 vLLM V1 的 ForwardContext。"""

    attn_metadata: AttentionMetadata


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    assert _forward_context is not None, "ForwardContext not set"
    return _forward_context


@contextmanager
def set_forward_context(attn_metadata: AttentionMetadata):
    global _forward_context
    _forward_context = ForwardContext(attn_metadata)
    try:
        yield
    finally:
        _forward_context = None
