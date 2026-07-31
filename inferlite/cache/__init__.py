"""KV Cache 数据结构与管理。

三代 KV Cache 设计：
  - M2 kv_cache.py:        单序列静态预分配 KV Cache
  - M3 batched_kv_cache.py: 固定 slot 池 + SlotManager（continuous batching）
  - M4 block_pool.py:       物理 block 池 + 引用计数（PagedAttention）
  - M4 paged_kv_cache.py:   block table 位置映射 + 分页 KV 容器
"""

from inferlite.cache.batched_kv_cache import (
    BatchedKVCache,
    BatchedLayerKVCache,
    SlotManager,
)
from inferlite.cache.block_pool import Block, BlockPool
from inferlite.cache.kv_cache import KVCache, LayerKVCache
from inferlite.cache.paged_kv_cache import (
    BlockTable,
    PagedKVCache,
    PagedLayerKVCache,
)

__all__ = [
    "KVCache",
    "LayerKVCache",
    "BatchedKVCache",
    "BatchedLayerKVCache",
    "SlotManager",
    "Block",
    "BlockPool",
    "BlockTable",
    "PagedKVCache",
    "PagedLayerKVCache",
]
