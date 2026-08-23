"""CacheAdapter：统一 M2/M3/M4 三种 cache 策略的适配层。

对齐 vLLM V1 的 KVCacheManager 模式，但更简化：
- vLLM V1 只有一条路径（paged），cache 管理在 scheduler 层
- inferlite 有 3 条路径（单序列 / batched slot / paged block），
  通过 CacheAdapter Protocol 统一接口，让 engine loop 不关心具体 cache 实现

设计要点：
- bind_kv_cache(model): 初始化时把 cache 绑定到每层 Attention，对齐 vLLM 的 kv_cache 直接赋值
- prepare_decode(ids): decode forward 前分配 cache 空间（仅 PagedCacheAdapter 实际做事）
- make_*_metadata(): 纯函数，从 cache 状态构造 AttentionMetadata
- can_admit / allocate / free: 生命周期管理

文件结构：
  CacheAdapter (Protocol)  → 公共接口
  SingleCacheAdapter       → M2: 单序列 KVCache，prefill + decode
  BatchedCacheAdapter      → M3: BatchedKVCache，fixed-slot continuous batching
  PagedCacheAdapter        → M4: PagedKVCache，paged block continuous batching
"""

from typing import Protocol

import torch

from inferlite.cache.batched_kv_cache import BatchedKVCache
from inferlite.cache.kv_cache import KVCache
from inferlite.cache.paged_kv_cache import PagedKVCache
from inferlite.engine.context import AttentionMetadata


# ── 1. CacheAdapter Protocol ──
# 公共接口定义：所有 adapter 必须实现这 8 个方法。
# engine loop 只通过这个接口操作 cache，不关心底层是无 cache / slot / block。
class CacheAdapter(Protocol):
    """Engine loop 与 cache 实现之间的适配层。
    对齐 vLLM V1 的 KVCacheManager 接口（简化版）。
    decode 时序由 prepare_decode() 处理，make_decode_metadata() 为纯函数。
    """

    # ── 生命周期 ──
    # can_admit: 检查 cache 是否还有容量接收新请求（slot 数 / block 数）
    def can_admit(self, prompt_len: int) -> bool: ...
    # allocate: 为新请求分配 cache 空间（M1 空操作，M2 reset，M3 分配 slot，M4 分配 block）
    def allocate(self, request_id: str, prompt_len: int) -> None: ...
    # free: 请求完成时释放 cache 空间
    def free(self, request_id: str) -> None: ...

    # ── decode 准备 ──
    # prepare_decode: forward 前调用，为每个请求分配新 token 的 cache 空间。
    # 对齐 vLLM V1 scheduler.allocate_slots() 的角色。
    def prepare_decode(self, request_ids: list[str]) -> None: ...

    # ── 元数据构造（纯函数，不修改 cache 状态）──
    def make_prefill_metadata(self, input_ids, positions) -> AttentionMetadata: ...
    def make_decode_metadata(self, next_tokens, positions) -> AttentionMetadata: ...

    # ── cache 绑定 ──
    # bind_kv_cache: 初始化时把 cache 绑定到模型的每个 Attention 层。
    # 对齐 vLLM V1 worker/utils.py 中 forward_context[layer_name].kv_cache = kv_cache
    def bind_kv_cache(self, model) -> None: ...


# ── 2. SingleCacheAdapter (M2) ──
# M2 模式：单序列 KVCache，prefill 一次写入全部 prompt 的 KV，
# decode 每步追加 1 个 token 的 KV，历史 KV 从 cache 读取（不重新计算）。
class SingleCacheAdapter:
    """M2 单序列 cache adapter。

    对应 core.py generate() 的 M2 路径：
    - prefill: 一次处理 prompt，KV 写入 cache
    - decode: 每步 1 token，历史 KV 从 cache 读

    bind_kv_cache → 每层 Attention.kv_cache = cache.layers[i] (LayerKVCache)
    cur_len 跟踪当前已写入的 token 数（与 cache.cur_len 保持同步）
    """

    def __init__(self, cache: KVCache):
        self.cache = cache  # KVCache 实例，包含 layers[i].k/v 各层缓存
        self.cur_len = 0  # 当前已写入 token 数（adapter 自维护，与 cache.cur_len 同步）

    def can_admit(self, prompt_len: int) -> bool:
        return True  # M2 单序列，无并发容量限制

    def bind_kv_cache(self, model) -> None:
        # 把 KVCache.layers[i] 绑定到第 i 层 Attention
        # → Attention.forward 检测 kv_cache is LayerKVCache → 走 M2 cache 路径
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.attn.kv_cache = self.cache.layers[i]
            layer.self_attn.attn.layer_idx = i

    def prepare_decode(self, request_ids: list[str]) -> None:
        pass  # M2 的 LayerKVCache 有预分配空间，不需要额外分配

    def make_prefill_metadata(self, input_ids, positions):
        # prefill 阶段：seq_len = prompt 长度 T
        # Attention 层会根据 seq_lens 知道要写多少 KV 到 cache
        T = input_ids.shape[1]
        return AttentionMetadata(
            num_seqs=1,
            seq_lens=torch.tensor([T]),
            cur_len=T,  # Python int，避免 .item() 同步
        )

    def make_decode_metadata(self, next_tokens, positions):
        # decode 阶段：cur_len += 1（当前步写入后的总长度）
        # 同步到 cache.cur_len，让 Attention 层知道 cache 里有多少历史 KV
        self.cur_len += 1
        self.cache.cur_len = self.cur_len
        return AttentionMetadata(
            num_seqs=1,
            seq_lens=torch.tensor([self.cur_len]),
            cur_len=self.cur_len,  # Python int，避免 .item() 同步
        )

    def allocate(self, request_id, prompt_len):
        # 新请求到来时：重置 cache（清空旧 KV），重置计数器
        self.cache.reset()
        self.cur_len = 0

    def free(self, request_id):
        pass  # M2 cache 在 generate() 结束时自然释放

    def set_seq_lens(self, requests) -> None:
        pass  # M2: cur_len 由 make_decode_metadata 管理

    # 在 SingleCacheAdapter 和 BatchedCacheAdapter 中加：
    def can_admit_with_cache(self, prompt_ids):
        return 0  # 不支持 prefix cache

    def allocate_with_cache(self, request_id, prompt_ids, num_cached):
        raise NotImplementedError  # 不会被调到（因为 can_admit_with_cache 返回 0）


# ── 4. BatchedCacheAdapter (M3) ──
# M3 模式：BatchedKVCache，固定 S 个 slot，每个 slot 存 max_seq_len 个 token。
# 支持 continuous batching：多个请求各占一个 slot，并行 decode。
class BatchedCacheAdapter:
    """M3 batched cache adapter。

    对应 batch_core.py batch_generate() 的主循环逻辑：
    - allocate: cache.allocate_slot() 分配一个 slot（slot_id 由 cache 内部管理）
    - prefill: 整段 prompt 写入对应 slot 的 [0, T) 位置
    - decode: 每步 1 token 写入 slot 的 [seq_len] 位置

    bind_kv_cache → 每层 Attention.kv_cache = cache.layers[i] (BatchedLayerKVCache)
    _current_request_ids: 跟踪当前 running 的请求 ID 列表（用于构造 metadata）
    """

    def __init__(self, cache: BatchedKVCache):
        self.cache = cache  # BatchedKVCache 实例
        self._current_request_ids: list[str] = []  # 当前 running 的请求 ID

    def can_admit(self, prompt_len: int) -> bool:
        # M3 容量 = 空闲 slot 数，有空 slot 就能 admit
        return len(self.cache.slot_manager.free_slots) > 0

    def bind_kv_cache(self, model) -> None:
        # BatchedLayerKVCache 和 LayerKVCache 结构相同，只是第一维是 S（slot 数）而非 1
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.attn.kv_cache = self.cache.layers[i]
            layer.self_attn.attn.layer_idx = i

    def prepare_decode(self, request_ids: list[str]) -> None:
        # M3: 同步 cache.seq_lens 到请求的实际 seq_len
        # 这样 make_decode_metadata 读取的 seq_lens 就是正确的「写入后长度」
        for rid in request_ids:
            slot = self.cache.slot_manager.req_to_slot[rid]
            # seq_lens 应该已经由外部（loop.py）更新到最新值，
            # 这里 +1 对齐旧 batch_core 的 cache_positions+1 语义
            self.cache.seq_lens[slot] += 1

    def set_seq_lens(self, requests) -> None:
        """Prefill 后同步请求的 seq_len 到 cache.seq_lens。

        Args:
            requests: RequestState 列表，每个需有 request_id 和 seq_len
        """
        for req in requests:
            slot = self.cache.slot_manager.req_to_slot[req.request_id]
            self.cache.seq_lens[slot] = req.seq_len

    def allocate(self, request_id, prompt_len):
        # 从 SlotManager 分配一个空闲 slot，记录 request_id → slot_id 映射
        self.cache.allocate_slot(request_id)
        self._current_request_ids.append(request_id)

    def free(self, request_id):
        # 释放 slot 回空闲池，从跟踪列表移除
        self.cache.free_slot(request_id)
        self._current_request_ids.remove(request_id)

    def make_prefill_metadata(self, input_ids, positions, request_ids=None):
        # prefill：支持多请求（batched prefill）
        # request_ids: 显式传入请求 ID 列表；不传时用 _current_request_ids 的最后 B 个
        B = input_ids.shape[0]
        if request_ids is None:
            request_ids = self._current_request_ids[-B:]
        T = input_ids.shape[1]
        slots = torch.tensor([self.cache.slot_manager.req_to_slot[rid] for rid in request_ids])
        # seq_lens: 每个请求的实际 prompt 长度（非 padded 长度）
        # 当所有请求 prompt 长度相同时，直接用 T；否则需要从 request 获取
        # 这里简化：用 positions 中每行的实际非零长度
        if positions.dim() == 2:
            # positions [B, T]: 每行的有效长度 = 最后一个非零位置 + 1
            seq_lens = []
            for i in range(B):
                nonzero = positions[i].nonzero(as_tuple=True)[0]
                if len(nonzero) > 0:
                    seq_lens.append(int(nonzero[-1]) + 1)
                else:
                    seq_lens.append(1)  # 单 token prompt
            seq_lens = torch.tensor(seq_lens)
        else:
            seq_lens = torch.full((B,), T)
        return AttentionMetadata(
            num_seqs=B,
            seq_lens=seq_lens,
            slot_mapping=slots,
        )

    def make_decode_metadata(self, next_tokens, positions):
        # decode：B 个请求并行
        # slot_mapping: 每个请求的 slot_id
        B = next_tokens.shape[0]
        slots = torch.tensor(
            [self.cache.slot_manager.req_to_slot[rid] for rid in self._current_request_ids]
        )
        seq_lens = self.cache.seq_lens[slots]  # [B] 各 slot 的当前序列长度
        return AttentionMetadata(
            num_seqs=B,
            seq_lens=seq_lens,
            slot_mapping=slots,
        )

    # 在 SingleCacheAdapter 和 BatchedCacheAdapter 中加：
    def can_admit_with_cache(self, prompt_ids):
        return 0  # 不支持 prefix cache

    def allocate_with_cache(self, request_id, prompt_ids, num_cached):
        raise NotImplementedError  # 不会被调到（因为 can_admit_with_cache 返回 0）


# ── 5. PagedCacheAdapter (M4) ──
# M4 模式：PagedKVCache，按 block 粒度管理 KV 存储。
# 每个 block 存 block_size 个 token 的 KV，请求按需分配 block（类似操作系统的分页内存）。
# 优势：消除 M3 fixed-slot 的内部碎片（不需要为每个请求预分配 max_seq_len 空间）。
class PagedCacheAdapter:
    """M4 paged cache adapter。

    对应 paged_core.py batch_generate_paged() 的主循环逻辑：
    - allocate: paged_cache.allocate_request() 按 prompt 长度分配初始 block
    - prefill: batched prefill，多请求 pad 到最长，一次 forward
    - decode: prepare_decode() 调 append_token() 按需追加 block
    - free: free_request() 释放所有 block 回 BlockPool

    bind_kv_cache → 每层 Attention.kv_cache = paged_cache（整个 cache 实例），layer_idx = i
    """

    def __init__(self, cache: PagedKVCache):
        self.cache = cache  # PagedKVCache 实例
        self._current_request_ids: list[str] = []  # 当前 running 的请求 ID

    def can_admit(self, prompt_len: int) -> bool:
        # M4 容量 = BlockPool 是否有足够空闲 block 容纳 prompt_len 个 token
        return self.cache.can_allocate(prompt_len)

    def can_admit_with_cache(self, prompt_ids: list[int]) -> int:
        """返回 prefix cache 命中 block 数。-1 = 容量不够，0 = 无命中。"""
        return self.cache.block_pool.can_allocate(prompt_ids)

    def bind_kv_cache(self, model) -> None:
        # PagedKVCache 是全局共享的（不是 per-layer），所以每层都绑定同一个 cache 实例
        # 通过 layer_idx 区分各层在 cache 中的写入位置
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.attn.kv_cache = self.cache
            layer.self_attn.attn.layer_idx = i

    def prepare_decode(self, request_ids: list[str]) -> None:
        # decode forward 前：为每个请求的下一个 token 分配 cache 空间
        for rid in request_ids:
            self.cache.append_token(rid)

    def set_seq_lens(self, requests) -> None:
        pass  # M4: seq_lens 由 block_table.seq_len 管理，不需要额外同步

    def allocate(self, request_id, prompt_len):
        # 新请求到来时：按 prompt 长度分配初始 block（decode 阶段按需追加）
        self.cache.allocate_request(request_id, prompt_len)
        self._current_request_ids.append(request_id)

    def allocate_with_cache(self, request_id: str, prompt_ids: list[int], num_cached: int) -> None:
        """cache-aware 分配。"""
        self.cache.allocate_request_with_cache(request_id, prompt_ids, num_cached)
        self._current_request_ids.append(request_id)

    def free(self, request_id):
        # 请求完成时：释放所有 block 回 BlockPool
        self.cache.free_request(request_id)
        if request_id in self._current_request_ids:
            self._current_request_ids.remove(request_id)

    def make_prefill_metadata(self, input_ids, positions, request_ids=None):
        # batched prefill：多个请求 pad 到最长 prompt 长度
        B = input_ids.shape[0]
        # 优先使用显式传入的 request_ids，否则用 _current_request_ids 最后 B 个
        if request_ids is None:
            request_ids = self._current_request_ids[-B:]
        # 从 BlockTable 获取每个请求的实际 prompt 长度
        prompt_lens = [self.cache.block_tables[rid].seq_len for rid in request_ids]
        seq_lens = torch.tensor(prompt_lens, dtype=torch.long)
        return AttentionMetadata(
            num_seqs=B,
            seq_lens=seq_lens,
            block_table=self._build_block_table(request_ids),
        )

    def make_decode_metadata(self, next_tokens, positions):
        # 纯函数：prepare_decode 已经 append_token 了，这里只读 cache 状态
        request_ids = self._current_request_ids
        B = len(request_ids)
        # 从 BlockTable 读取每个请求 append 后的序列长度
        seq_lens = torch.tensor([self.cache.block_tables[rid].seq_len for rid in request_ids])
        return AttentionMetadata(
            num_seqs=B,
            seq_lens=seq_lens,
            block_table=self._build_block_table(request_ids),
        )

    def _build_block_table(self, request_ids):
        """从 PagedKVCache 提取 block_table tensor。

        将每个请求的 block_ids 列表拼成 [num_seqs, max_blocks] 的 2D tensor，
        短的用 0 padding（Attention 层通过 seq_lens 知道有效范围）。

        Returns:
            [num_seqs, max_blocks] int tensor
        """
        tables = [self.cache.block_tables[rid].block_ids for rid in request_ids]
        max_blocks = max(len(t) for t in tables)
        block_table = torch.zeros(len(request_ids), max_blocks, dtype=torch.long)
        for i, t in enumerate(tables):
            block_table[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        return block_table
