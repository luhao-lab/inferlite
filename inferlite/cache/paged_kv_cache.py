"""M4 PagedAttention 的分页 KV Cache。

本文件承载两个协作但职责分离的组件：

  - BlockTable（T2）：单请求的 pos -> (physical_block_id, block_offset) 映射。
    映射本身只做整数运算，不依赖 torch 或 BlockPool。
  - PagedKVCache（T3）：持有各层 K/V tensor，组合 BlockPool 与 BlockTable，
    负责请求生命周期、批量 KV scatter 与 gather。

分层关系：BlockPool 管「块归谁」（全局一个），BlockTable 管「块的顺序」
（每请求一个）。两者互不认识，由 PagedKVCache 组合。

关于 block_id 的语义：它是一个**编号**，不是内存地址。T3 以它作为 K/V
物理 tensor 的第 0 维下标；`slot = block_id * block_size + offset` 是其展平
后的等价坐标。
"""

from dataclasses import dataclass, field
from math import ceil

import torch

from inferlite.cache.block_pool import BlockPool
from inferlite.config import ModelConfig


@dataclass
class BlockTable:
    """单个请求的逻辑 block 到物理 block 的映射。

    请求看到连续的逻辑地址 pos = 0..seq_len-1，底层可以落在完全非连续、
    甚至倒序的物理 block 上。这层间接映射是 PagedAttention 能消除内存
    碎片的原因：不必为每个请求预留一整段连续的 max_seq_len 空间。

    block_ids 的**下标**是 logical block index，**值**是 physical block id，
    两者不可混淆。block_ids=[7, 2] 表示逻辑第 0 块落在物理块 7、逻辑第 1 块
    落在物理块 2。所以它必须是有序 list 而非 set —— 顺序本身就是信息。

    状态不变量：
    - 0 <= seq_len <= capacity == len(block_ids) * block_size
    - position_to_block 的合法定义域为 [0, seq_len)
    - num_full_blocks * block_size + last_block_offset == seq_len
    - block_ids 中所有元素非负

    seq_len 和 block_ids 应只通过 extend() / append_block() 修改：这两个写
    入口负责守住 seq_len <= capacity。字段本身未做私有化（保持 dataclass
    构造与测试的简洁），因此这是约定而非强制 —— 直接赋值 seq_len 会破坏
    不变量，进而让 position_to_block 返回其他请求的物理块。

    M4 不含 prefix caching，故不预埋 block_hashes / token_ids 字段（留 M5）。
    """

    request_id: str
    block_size: int
    # 逻辑块顺序 -> 物理块编号。必须用 default_factory，否则所有实例会共享
    # 同一个 list，多请求之间彼此串数据。
    block_ids: list[int] = field(default_factory=list)
    seq_len: int = 0

    def __post_init__(self) -> None:
        """校验构造出的初始状态。

        dataclass 自动生成的 __init__ 无法插入校验逻辑，__post_init__ 是它在
        所有字段赋值完成后调用的钩子，因此这里可以安全读取 self.capacity。

        这些都是调用方可能触发的参数错误，所以用显式 raise 而非 assert
        （python -O 会移除 assert，让校验在生产环境静默失效）。
        """
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.seq_len < 0:
            raise ValueError(f"seq_len must be non-negative, got {self.seq_len}")
        if any(bid < 0 for bid in self.block_ids):
            raise ValueError("block_ids must all be non-negative")
        # capacity 依赖 block_size，故必须放在 block_size 校验之后，否则
        # block_size=-4 会算出负 capacity，让这条检查失去意义。
        # 用 > 而非 >=：seq_len == capacity 表示「正好写满」，是合法状态。
        if self.seq_len > self.capacity:
            raise ValueError(f"seq_len {self.seq_len} exceeds capacity {self.capacity}")

    # ── 只读视图 ──
    # 这四个量都是 block_ids / seq_len 的派生值，故用 property 实时计算而非
    # 存成字段：存字段就会出现「两份事实」，append_block 后忘记同步就不一致。

    @property
    def num_blocks(self) -> int:
        """当前持有的物理 block 数（已分配，不代表已写满）。"""
        return len(self.block_ids)

    @property
    def capacity(self) -> int:
        """已分配空间能容纳的 token 上限，也是 seq_len 的合法上界。"""
        return self.block_size * self.num_blocks

    @property
    def num_full_blocks(self) -> int:
        """已被写满的 block 数。"""
        return self.seq_len // self.block_size

    @property
    def last_block_offset(self) -> int:
        """最后一个 block 中**已用**的 token 数。

        注意恰好写满时返回 0 而不是 block_size。误当成「还剩 0 个位置」
        就会漏分配，判断是否需要新块只用 needs_new_block()。
        """
        return self.seq_len % self.block_size

    # ── 查询 ──

    def needs_new_block(self) -> bool:
        """再追加 1 个 token 之前，是否需要先分配新 block。

        直接比较 seq_len 与 capacity，而不是 seq_len % block_size == 0：
        后者在 capacity == 0（请求刚创建、还没拿到任何块）时漏判返回 False，
        且在 prefill 预分配了富余块时误判返回 True，导致上层白白多要一块。
        """
        return self.seq_len >= self.capacity

    def position_to_block(self, pos: int) -> tuple[int, int]:
        """把逻辑 token 位置翻译成 (physical_block_id, block_offset)。"""
        # 必须显式拒绝越界，尤其是负数：pos=-1 会让 block_ids[-1] 静默返回
        # 最后一个物理块，不报错却读到无关数据。pos >= seq_len 则会读到从未
        # 写入的位置，在 T3 里是 torch.empty 的未初始化内存（可能含 NaN）。
        if not 0 <= pos < self.seq_len:
            raise ValueError(f"pos {pos} out of range [0, {self.seq_len})")
        logical_block = pos // self.block_size
        block_offset = pos % self.block_size
        # 内部不变量：pos < seq_len <= capacity == num_blocks * block_size
        # 已保证此条成立，触发则说明不变量维护逻辑有 bug，而非调用方错误，
        # 故用 assert 而非 raise。
        assert logical_block < self.num_blocks
        return self.block_ids[logical_block], block_offset

    # ── 受控写入口 ──

    def append_block(self, physical_block_id: int) -> None:
        """追加一个物理 block，capacity 增加 block_size。

        刻意**不改动 seq_len**：分配空间和写入 token 是两件事，分开才能
        表达「已分配但未写满」的中间状态。

        只校验非负，不校验上界 physical_block_id < num_blocks —— BlockTable
        刻意不依赖 BlockPool，因此拿不到 num_blocks。上界由 BlockPool 在
        allocate() 发号时保证，它才是知道总量的那一方。这不是遗漏。
        """
        if physical_block_id < 0:
            raise ValueError(f"physical_block_id must be non-negative, got {physical_block_id}")
        self.block_ids.append(physical_block_id)

    def extend(self, num_tokens: int) -> None:
        """声明已写入 num_tokens 个 token。prefill 传 prompt_len，decode 传 1。"""
        if num_tokens < 0:
            raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
        # 容量不足说明调用方没有先 append_block()，属可预期的调用顺序错误，
        # 与 T1 BlockPool.allocate() 无空闲块时抛 RuntimeError 保持一致。
        # 先检查后修改，保证失败时 seq_len 不留半改状态。
        if self.seq_len + num_tokens > self.capacity:
            raise RuntimeError(
                f"cannot extend {num_tokens} tokens: "
                f"seq_len {self.seq_len} + {num_tokens} > capacity {self.capacity}"
            )
        self.seq_len += num_tokens


@dataclass
class PagedLayerKVCache:
    """单层 paged KV 数据容器。

    k/v shape: [num_blocks, block_size, n_kv_heads, head_dim]
    """

    k: torch.Tensor
    v: torch.Tensor


class PagedKVCache:
    """多层分页 KV 容器。

    `block_pool` 管全局物理 block 的分配与回收；`block_tables` 为每个活跃
    请求保存逻辑位置到物理 block 的映射；本类则持有真实 K/V tensor 并将
    batch 的连续 K/V 输入 scatter 到不连续物理 slot，或反向 gather 回连续序列。

    M4 中 block 在请求之间独占。M5 prefix cache 引入共享后，写共享 block 前
    必须在本类完成 Copy-on-Write，不能直接沿用当前的原地 scatter。
    """

    def __init__(
        self,
        layers: list[PagedLayerKVCache],
        block_pool: BlockPool,
        block_size: int,
    ) -> None:
        self.layers = layers
        self.block_pool = block_pool
        self.block_size = block_size
        self.block_tables: dict[str, BlockTable] = {}

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "PagedKVCache":
        layers: list[PagedLayerKVCache] = []
        for _ in range(config.num_hidden_layers):
            k = torch.empty(
                num_blocks,
                block_size,
                config.num_key_value_heads,
                config.head_dim,
                dtype=dtype,
                device=device,
            )
            v = torch.empty(
                num_blocks,
                block_size,
                config.num_key_value_heads,
                config.head_dim,
                dtype=dtype,
                device=device,
            )
            layers.append(PagedLayerKVCache(k=k, v=v))
        return cls(layers, BlockPool(num_blocks, block_size), block_size)

    # —— 请求生命周期 ——
    def allocate_request(self, request_id: str, prompt_len: int) -> None:
        if request_id in self.block_tables:
            raise ValueError(f"request_id {request_id} already allocated")
        num_needed = self._blocks_needed(prompt_len)

        table = BlockTable(request_id=request_id, block_size=self.block_size)
        allocated: list[int] = []
        try:
            for _ in range(num_needed):
                block_id = self.block_pool.allocate()  # 池空时抛 RuntimeError
                allocated.append(block_id)
                table.append_block(physical_block_id=block_id)
            table.extend(num_tokens=prompt_len)
        except RuntimeError:
            # 回滚：把本次已拿到的块全部归还，否则永久泄漏
            for block_id in allocated:
                self.block_pool.free(block_id)
            raise  # 重新抛出异常，让调用方知道分配失败

        self.block_tables[request_id] = table

    def allocate_request_with_cache(
        self, request_id: str, token_ids: list[int], num_cached: int
    ) -> None:
        """cache-aware 分配：touch cached block + allocate 新 block。"""
        if request_id in self.block_tables:
            raise ValueError(f"request_id {request_id} already allocated")
        block_ids = self.block_pool.allocate_with_cache(token_ids, num_cached)
        table = BlockTable(request_id=request_id, block_size=self.block_size)
        for bid in block_ids:
            table.append_block(physical_block_id=bid)
        table.extend(num_tokens=len(token_ids))
        self.block_tables[request_id] = table

    def append_token(self, request_id: str) -> None:
        table = self.block_tables[request_id]
        if table.needs_new_block():
            block_id = self.block_pool.allocate()
            table.append_block(physical_block_id=block_id)
        table.extend(1)

    def free_request(self, request_id: str) -> None:
        table = self.block_tables.pop(request_id)
        for block_id in table.block_ids:
            self.block_pool.free(block_id)

    # —— KV 写入 ——
    def write_prefill(
        self, layer_idx: int, request_ids: list[str], k: torch.Tensor, v: torch.Tensor
    ) -> None:
        if not request_ids:
            raise ValueError("request_ids must not be empty")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_ids must not contain duplicates")
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError(f"layer_idx {layer_idx} out of range")
        if k.shape != v.shape:
            raise ValueError("k and v must have identical shapes")
        if k.shape[0] != len(request_ids):
            raise ValueError("batch size must match request_ids")
        if k.ndim != 4:
            raise ValueError("write_prefill expects k/v shape [B, n_kv_heads, T_max, head_dim]")

        tables = [self.block_tables[request_id] for request_id in request_ids]
        # table.seq_len 在 allocate_request() 时由 prompt_len 初始化；
        # prefill 阶段它就是每个 batch 行的有效 token 数。
        seq_lens = [table.seq_len for table in tables]
        t_max = k.shape[2]

        if t_max < max(seq_lens):
            raise ValueError("T_max is shorter than a request seq_len")

        # 2. 地址与数据按完全相同的「请求优先、位置递增」顺序展平
        layer: PagedLayerKVCache = self.layers[layer_idx]

        expected_shape = (layer.k.shape[2], layer.k.shape[3])
        if k.shape[1] != expected_shape[0] or k.shape[3] != expected_shape[1]:
            raise ValueError(
                f"expected [B, {expected_shape[0]}, T, {expected_shape[1]}], got {tuple(k.shape)}"
            )

        if k.dtype != layer.k.dtype or k.device != layer.k.device:
            raise ValueError("k/v dtype and device must match cache layer")

        # [total_valid_tokens]，例如请求长度 [20, 5] 时长度是 25。
        slot_mapping = self._make_prefill_slot_mapping(request_ids, layer.k.device)

        positions = torch.arange(t_max, device=k.device)
        lengths = torch.tensor(seq_lens, device=k.device)
        valid = positions[None, :] < lengths[:, None]
        flat_k = k.transpose(1, 2)[valid]
        flat_v = v.transpose(1, 2)[valid]

        assert slot_mapping.numel() == flat_k.shape[0] == flat_v.shape[0]

        # 3. 一次 batch scatter
        flat_cache_k = layer.k.view(-1, *layer.k.shape[2:])
        flat_cache_v = layer.v.view(-1, *layer.v.shape[2:])
        flat_cache_k[slot_mapping] = flat_k
        flat_cache_v[slot_mapping] = flat_v

    def write_decode(
        self, layer_idx: int, request_ids: list[str], k: torch.Tensor, v: torch.Tensor
    ) -> None:
        if not request_ids:
            raise ValueError("request_ids must not be empty")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_ids must not contain duplicates")
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError(f"layer_idx {layer_idx} out of range")
        if k.shape != v.shape:
            raise ValueError("k and v must have identical shapes")
        if k.shape[0] != len(request_ids):
            raise ValueError("batch size must match request_ids")
        if k.ndim != 4 or k.shape[2] != 1:
            raise ValueError("write_decode expects k/v shape [B, n_kv_heads, 1, head_dim]")

        layer: PagedLayerKVCache = self.layers[layer_idx]
        expected_shape = (layer.k.shape[2], layer.k.shape[3])
        if k.shape[1] != expected_shape[0] or k.shape[3] != expected_shape[1]:
            raise ValueError(
                f"expected [B, {expected_shape[0]}, 1, {expected_shape[1]}], got {tuple(k.shape)}"
            )
        if k.dtype != layer.k.dtype or k.device != layer.k.device:
            raise ValueError("k/v dtype and device must match cache layer")

        # [B]；调用方必须先 append_token()，使每个 table 的 seq_len 指向
        # 本轮刚新增 token 的后一位，helper 再以 seq_len - 1 定位其 slot。
        slot_mapping = self._make_decode_slot_mapping(request_ids, layer.k.device)

        # 一次 batch scatter
        flat_cache_k = layer.k.view(-1, *layer.k.shape[2:])
        flat_cache_v = layer.v.view(-1, *layer.v.shape[2:])
        flat_cache_k[slot_mapping] = k.squeeze(2)
        flat_cache_v[slot_mapping] = v.squeeze(2)

    # —— Attenion 读取 ——
    def gather_kv(
        self, layer_idx: int, request_ids: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量读取分页 KV，返回 block 对齐 padding 与每行有效长度。

        注意：padding 区可能含 torch.empty 的 NaN/Inf。调用方必须根据
        valid_lens 清零 K/V，再进行 attention。
        """
        if not request_ids:
            raise ValueError("request_ids must not be empty")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_ids must not contain duplicates")
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError(f"layer_idx {layer_idx} out of range")

        tables = [self.block_tables[request_id] for request_id in request_ids]
        layer = self.layers[layer_idx]

        batch_size = len(tables)
        max_num_blocks = max(table.num_blocks for table in tables)

        # 将变长 block_ids 补齐成矩阵：
        #
        # a: [7, 2]
        # b: [5]
        #
        # block_table:
        # [[7, 2],
        #  [5, 0]]    # 0 只是 padding 占位，后续必须由 valid_lens 屏蔽
        block_table = torch.zeros(
            (batch_size, max_num_blocks),
            dtype=torch.long,
            device=layer.k.device,
        )
        for batch_idx, table in enumerate(tables):
            block_table[batch_idx, : table.num_blocks] = torch.tensor(
                data=table.block_ids,
                dtype=torch.long,
                device=layer.k.device,
            )
        # 一次高级索引：
        #
        # layer.k:             [num_blocks, block_size, n_kv, D]
        # layer.k[block_table]: [B, max_num_blocks, block_size, n_kv, D]
        k = layer.k[block_table]
        v = layer.v[block_table]

        # 合并「逻辑 block」与「block 内 offset」，再把 head 换到第 1 维，
        # 对齐 attention 的 K/V layout。
        #
        # [B, nb, bs, n_kv, D]
        # -> [B, nb * bs, n_kv, D]
        # -> [B, n_kv, L_pad, D]
        k = k.reshape(batch_size, max_num_blocks * self.block_size, *k.shape[3:])
        v = v.reshape(batch_size, max_num_blocks * self.block_size, *v.shape[3:])
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        valid_lens = torch.tensor(
            [table.seq_len for table in tables],
            dtype=torch.long,
            device=layer.k.device,
        )
        return k, v, valid_lens

    def gather_kv_single(
        self, layer_idx: int, request_id: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """读取一个请求的连续有效 KV，不含 padding，仅用于测试 oracle。"""
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError(f"layer_idx {layer_idx} out of range")

        table = self.block_tables[request_id]  # 未注册时自然抛 KeyError
        layer = self.layers[layer_idx]

        # 例如 table.block_ids = [7, 2]。
        # layer.k[block_ids]：
        # [num_blocks, block_size, n_kv, D]
        # -> [2, block_size, n_kv, D]
        block_ids = torch.tensor(
            table.block_ids,
            dtype=torch.long,
            device=layer.k.device,
        )

        # 逻辑顺序已由 block_ids 的列表顺序保证：
        # [block 7 的全部 token, block 2 的全部 token]
        # -> [num_blocks * block_size, n_kv, D]
        k = layer.k[block_ids].reshape(-1, *layer.k.shape[2:])
        v = layer.v[block_ids].reshape(-1, *layer.v.shape[2:])

        # 最后一个 block 通常没写满，必须截到真实有效长度。
        # [seq_len, n_kv, D] -> [n_kv, seq_len, D]
        k = k[: table.seq_len].transpose(0, 1)
        v = v[: table.seq_len].transpose(0, 1)
        return k, v

    # -- 容量查询（T5 admission 用）
    @property
    def num_free_blocks(self) -> int:
        """当前空闲的物理 block 数。"""
        return self.block_pool.num_free_blocks

    def can_allocate(self, prompt_len: int) -> bool:
        """检查能否为指定长度的 prompt 分配空间。"""
        return self.block_pool.can_allocate(self._blocks_needed(prompt_len))

    def seq_len_of(self, request_id: str) -> int:
        return self.block_tables[request_id].seq_len

    # —— 校验 ——
    def _blocks_needed(self, prompt_len: int) -> int:
        """把 prompt 长度换算成需要的块数，顺带校验。

        allocate_request 和 can_allocate 都经由此处，校验逻辑只有一份。
        """
        if prompt_len <= 0:
            raise ValueError(f"prompt_len must be positive, got {prompt_len}")
        return ceil(prompt_len / self.block_size)

    def _make_prefill_slot_mapping(
        self,
        request_ids: list[str],
        device: torch.device | str,
    ) -> torch.Tensor:
        """按 request_ids 顺序生成所有有效 prefill token 的物理 slot。

        返回 slot_mapping[i] 与 write_prefill 中 flat_k[i] / flat_v[i]
        一一对应，顺序均为「请求优先、请求内位置递增」。
        """
        slots_per_request: list[torch.Tensor] = []
        offsets = torch.arange(self.block_size, device=device)

        for request_id in request_ids:
            table = self.block_tables[request_id]

            # [num_blocks, 1] * block_size + [block_size]
            # -> [num_blocks, block_size]
            block_ids = torch.tensor(table.block_ids, dtype=torch.long, device=device)
            slots = block_ids[:, None] * self.block_size + offsets[None, :]

            # flatten 后是：
            # block 0 的 offset 0..block_size-1，
            # 再 block 1 的 offset 0..block_size-1，顺序正好对应逻辑 token 顺序。
            slots_per_request.append(slots.flatten()[: table.seq_len])

        return torch.cat(slots_per_request)

    def _make_decode_slot_mapping(
        self,
        request_ids: list[str],
        device: torch.device | str,
    ) -> torch.Tensor:
        """按 request_ids 顺序生成所有有效 decode token 的物理 slot。

        返回 slot_mapping[i] 与 write_decode 中 k[i] / v[i]
        一一对应，顺序均为「请求优先、请求内位置递增」。
        """
        slots_per_request: list[torch.Tensor] = []

        for request_id in request_ids:
            table: BlockTable = self.block_tables[request_id]
            block_id = table.block_ids[-1]  # 最后一个 block
            slot = block_id * self.block_size + ((table.seq_len - 1) % self.block_size)
            slots_per_request.append(torch.tensor([slot], device=device))

        return torch.cat(slots_per_request)
