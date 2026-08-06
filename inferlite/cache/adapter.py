from typing import Protocol

from inferlite.engine.forward_context import AttentionMetadata


# ── 1. CacheAdapter Protocol ──
# 公共接口定义
class CacheAdapter(Protocol):
    """Engine loop 与 cache 实现之间的适配层
    对齐 vLLM V1 的 KVCacheManager 接口。
    """

    # ── 生命周期 ──
    def can_admit(self, request) -> bool: ...
    def allocate(self, request) -> None: ...
    def free(self, request_id: str) -> None: ...

    # ── decode 三步时序 ──
    def prepare_decode(self, request_id: str) -> None: ...  # 分配空间（不 +1）
    def decode_positions(self, request_ids: list[str]) -> list[int]: ...  # 读 position
    def commit_decode(self, request_ids: list[str]) -> None: ...  # +1

    # ── 元数据构造 ──
    def make_prefill_metadata(self, requests) -> AttentionMetadata: ...
    def make_decode_metadata(self, requests) -> AttentionMetadata: ...

    # ── cache 绑定 ──
    def bind_kv_cache(self, model) -> None:
        """初始化时将 cache 绑定到模型的每个 Attention 层。
        对齐 vLLM V1 的 bind_kv_cache 模式。
        """
        ...
