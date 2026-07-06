"""LLM Model 模块。

导出：
  - BatchedKVCache: M3 固定槽位 KV Cache
  - BatchedLayerKVCache: 单层 KV 数据容器
  - SlotManager: slot 分配/释放管理
"""

from inferlite.model.batched_kv_cache import (
    BatchedKVCache,
    BatchedLayerKVCache,
    SlotManager,
)

__all__ = ["BatchedKVCache", "BatchedLayerKVCache", "SlotManager"]
