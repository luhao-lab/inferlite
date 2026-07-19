"""M3 Continuous Batching 的固定槽位 KV Cache。

与 M2 的 kv_cache.py 的区别：
  - M2: batch 维是"同步组"（全局 cur_len，所有请求锁步进退）
  - M3: slot 维是"独立请求"（per-slot seq_lens，每个请求独立进退）

本模块包含三个类：
  - BatchedLayerKVCache: 单层 KV 数据容器，shape [S, H_kv, L, D]
  - SlotManager: slot 分配/释放管理（deque + dict）
  - BatchedKVCache: 多层 cache + per-slot 元数据（seq_lens, occupied）

生命周期：
  1. from_config() 预分配固定大小的 slot 池
  2. allocate_slot() 为新请求分配 slot
  3. BatchEngine 在推理时直接写 cache.layers[i].k/v 并更新 seq_lens
  4. free_slot() 释放 slot（清元数据，不清 tensor）
  5. reset_slots() 全部清零（benchmark 两轮之间）
"""

from collections import deque
from dataclasses import dataclass

import torch

from inferlite.config import ModelConfig


@dataclass
class BatchedLayerKVCache:
    """单层 KV 数据容器。

    k/v shape: [S, n_kv_heads, max_seq_len, head_dim]
      - S: 固定槽位数（= max_num_slots）
      - n_kv_heads: GQA 的 KV 头数
      - max_seq_len: 预分配的最大序列长度
      - head_dim: 每个头的维度
    """

    k: torch.Tensor
    v: torch.Tensor


class SlotManager:
    """管理slot的分配和释放"""

    def __init__(self, max_num_slots: int) -> None:
        self.max_num_slots: int = max_num_slots
        self.free_slots: deque[int] = deque(range(max_num_slots))
        self.req_to_slot: dict[str, int] = {}

    def allocate(self, request_id: str) -> int:
        """分配一个 slot。
        - request_id 重复 → raise ValueError
        - free_slots 空 → raise RuntimeError（防御性检查）
        """
        if request_id in self.req_to_slot:
            raise ValueError(f"request_id {request_id} already allocated")
        elif not self.free_slots:
            raise RuntimeError("no free slots")
        else:
            slot_id = self.free_slots.popleft()
            self.req_to_slot[request_id] = slot_id
            return slot_id

    def free(self, request_id: str) -> None:
        """释放 slot。传入 request_id（调用方有 RequestState 对象）。
        - request_id 不在 req_to_slot → raise ValueError
        """
        if request_id not in self.req_to_slot:
            raise ValueError(f"request_id {request_id} not found")
        else:
            slot_id = self.req_to_slot.pop(request_id)
            self.free_slots.append(slot_id)

    def is_free(self, slot_id: int) -> bool:
        """slot_id 是否空闲。"""
        return slot_id in self.free_slots


class BatchedKVCache:
    """固定槽位 KV Cache，支持 continuous batching。

    与 M2 KVCache 的核心区别：
      - M2: 全局 cur_len（所有请求同步）
      - M3: per-slot seq_lens（每个请求独立进退）

    字段：
      - layers: 每层的 k/v tensor，shape [S, H_kv, L, D]
      - seq_lens: per-slot 当前有效长度，shape [S]
      - occupied: per-slot 是否被占用，shape [S]
      - slot_manager: slot 分配器
    """

    def __init__(
        self, layers: list[BatchedLayerKVCache], max_seq_len: int, max_num_slots: int, device=None
    ) -> None:
        self.layers = layers
        self.max_seq_len = max_seq_len
        self.max_num_slots = max_num_slots
        # seq_lens: 每个 slot 的当前有效长度（= prompt_len + num_generated）
        self.seq_lens = torch.zeros(max_num_slots, dtype=torch.long, device=device)
        # occupied: 每个 slot 是否被占用（和 SlotManager.req_to_slot 对应）
        self.occupied = torch.zeros(max_num_slots, dtype=torch.bool, device=device)
        self.slot_manager = SlotManager(max_num_slots)

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        max_num_slots: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "BatchedKVCache":
        """按模型配置创建固定槽位 cache。参考 M2 的 KVCache.from_config()。"""
        layers: list[BatchedLayerKVCache] = []
        for _ in range(config.num_hidden_layers):
            k = torch.empty(
                max_num_slots,
                config.num_key_value_heads,
                max_seq_len,
                config.head_dim,
                dtype=dtype,
                device=device,
            )
            v = torch.empty(
                max_num_slots,
                config.num_key_value_heads,
                max_seq_len,
                config.head_dim,
                dtype=dtype,
                device=device,
            )
            layers.append(BatchedLayerKVCache(k=k, v=v))
        return cls(layers, max_seq_len, max_num_slots, device=device)

    def reset_slots(self) -> None:
        """重置所有 slot 为可用状态。"""
        self.seq_lens.zero_()
        self.occupied.zero_()
        self.slot_manager = SlotManager(self.max_num_slots)

    def free_slot(self, request_id: str) -> None:
        """释放指定 request_id 占用的 slot。"""
        if request_id not in self.slot_manager.req_to_slot:
            raise ValueError(f"request_id {request_id} not allocated")
        slot_id = self.slot_manager.req_to_slot[request_id]
        self.occupied[slot_id] = False
        self.seq_lens[slot_id] = 0
        self.slot_manager.free(request_id)

    def allocate_slot(self, request_id: str) -> int:
        """为请求分配一个空闲 slot，并标记为已占用。"""
        slot_id = self.slot_manager.allocate(request_id)
        self.occupied[slot_id] = True
        return slot_id
