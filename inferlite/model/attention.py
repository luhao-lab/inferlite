"""Qwen3 Attention 模块：拆分为 Qwen3Attention（上层）和 Attention（下层）。

对齐 vLLM V1 的两层 attention 架构：
  Qwen3Attention: projection + QK-norm + RoPE + 委托 Attention + o_proj
  Attention:      cache 读写 + repeat_kv + scores + mask + softmax + matmul

Qwen3Attention.forward 签名对齐 vLLM V1 Qwen3Attention：
  forward(positions, hidden_states) -> hidden_states

Attention.forward 签名对齐 vLLM V1 Attention：
  forward(q, k, v) -> attn_output
  cache 通过 self.kv_cache（adapter 初始化时绑定）和 ForwardContext（运行时设置）获取。
Qwen3 GQA Attention 的最小数值对齐实现。

写 Attention 可以按下面这份伪代码拆：

1. 先从 config 固化结构超参
   - hidden_size: H，残差流宽度，也就是每个 token 在主干网络里的向量宽度
   - num_heads: n_q，Query head 数
   - num_key_value_heads: n_kv，Key/Value head 数；GQA 中它通常小于 n_q
   - head_dim: D，单 head 宽度，Qwen3-0.6B 显式给出 128，不能用 H / n_q 推导
   - num_key_value_groups: n_q / n_kv，GQA 中每个 KV head 被多少个 Q head 共享
   - scaling: D ** -0.5，attention score 缩放因子，避免 q·k 随 D 变大而过大

2. 再定义 Attention 子结构
   - q_proj / k_proj / v_proj: hidden_states -> q/k/v
   - o_proj: 多头 attention output -> hidden_size，回到 residual stream
   - q_norm / k_norm: Qwen3 特有，RoPE 前只在 head_dim 上做 RMSNorm
   - rotary_emb: 根据 position_ids 生成 cos/sin，真正旋转由 apply_rotary_pos_emb 完成

3. forward 按数据流写
   hidden_states [B, T, H]
     -> q/k/v projection
     -> reshape to [B, heads, T, D]
     -> q_norm / k_norm
     -> RoPE(q, k)
     -> repeat_kv(k, v)
     -> q @ k^T * scaling
     -> causal mask
     -> softmax
     -> attn @ v
     -> o_proj

T4 新增：M4 PagedAttention（分页 KV）路径。forward 现在支持四路：
  M1 无 cache / M2 单序列 cache / M3 batched fixed-slot cache / M4 paged cache。
"""

from typing import override

import torch
import torch.nn as nn

from inferlite.cache import BatchedLayerKVCache, PagedKVCache
from inferlite.cache.kv_cache import LayerKVCache
from inferlite.config import ModelConfig
from inferlite.engine.context import get_forward_context
from inferlite.model.layers import RMSNorm, RotaryEmbedding, apply_rotary_pos_emb


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 GQA 的 KV heads repeat 到 Query heads 数量。

    GQA（Grouped Query Attention）的核心是：
    - Query 头更多：`num_heads = n_q`
    - Key/Value 头更少：`num_key_value_heads = n_kv`
    - 多个 Query head 共享同一个 Key/Value head

    因此 attention 真正做 `q @ k^T` 前，需要把 k/v 从 n_kv 个 head 复制到 n_q 个 head：

        [B, n_kv, T, D] -> [B, n_kv, n_rep, T, D] -> [B, n_q, T, D]

    以 Qwen3-0.6B 为例：
    - n_q = 16
    - n_kv = 8
    - n_rep = 2

    所以第 0 个 KV head 会服务第 0/1 个 Query head，第 1 个 KV head 会服务
    第 2/3 个 Query head，以此类推。

    Args:
        hidden_states: [B, num_key_value_heads, T, head_dim]
        n_rep: 每个 KV head 复制给多少个 Query head 使用。

    Returns:
        [B, num_key_value_heads * n_rep, T, head_dim]
    """
    if n_rep == 1:
        # MHA 退化情况：如果 n_q == n_kv，就不需要 repeat。
        return hidden_states

    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape

    # 先在 KV head 后面插入一个 group 维度：
    #   [B, n_kv, T, D] -> [B, n_kv, 1, T, D]
    # 再用 expand 逻辑复制 n_rep 份。expand 不立刻拷贝数据，只创建广播视图。
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        n_rep,
        seq_len,
        head_dim,
    )

    # 最后把 [n_kv, n_rep] 合并成 n_q。reshape 会在需要时 materialize。
    return hidden_states.reshape(
        batch_size,
        num_key_value_heads * n_rep,
        seq_len,
        head_dim,
    )


# ── 1. Attention（下层）──
# 纯粹的 attention 计算：cache RW → repeat_kv → scores → mask → softmax → matmul。
# 不持有任何 projection 层（q/k/v/o_proj），不包含 QK-norm 和 RoPE。
class Attention(nn.Module):
    """底层 attention 计算模块。

    对齐 vLLM V1 的 Attention 类：
    - self.kv_cache: 由 adapter.bind_kv_cache() 在初始化时绑定
    - cache 元数据从 get_forward_context().attn_metadata 读取
    - forward(q, k, v): 纯计算，3 参数

    cache 路径选择（根据 self.kv_cache 类型）：
      None                 → M1: 标准 causal attention
      LayerKVCache         → M2: 单序列 cache RW
      BatchedLayerKVCache  → M3: batched slot cache RW
      PagedKVCache         → M4: paged block cache RW
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        # 运行时由 adapter.bind_kv_cache() 赋值
        self.kv_cache = None
        self.layer_idx: int | None = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """执行 attention 计算。

        Args:
            q: [B, n_q, T, D] — 已经过 QK-norm + RoPE
            k: [B, n_kv, T, D] — 已经过 QK-norm + RoPE
            v: [B, n_kv, T, D]

        Returns:
            [B, T, n_q * D] — attention output（未过 o_proj）
        """
        batch_size, _, seq_len, _ = q.shape

        # ── 1. KV Cache 读写 ──
        # 根据 self.kv_cache 类型选择路径
        paged_valid_lens: torch.Tensor | None = None
        cache_positions: torch.Tensor | None = None

        if self.kv_cache is None:
            pass  # M1: 直接用当前 q/k/v
        elif isinstance(self.kv_cache, PagedKVCache):
            # M4: paged cache RW
            metadata = get_forward_context().attn_metadata
            k, v, paged_valid_lens = self._paged_cache_rw(k, v, metadata)
        elif isinstance(self.kv_cache, BatchedLayerKVCache):
            # M3: batched slot cache RW
            metadata = get_forward_context().attn_metadata
            k, v, cache_positions = self._batched_cache_rw(k, v, metadata)
        elif isinstance(self.kv_cache, LayerKVCache):
            # M2: single cache RW
            metadata = get_forward_context().attn_metadata
            k, v, cache_position = self._single_cache_rw(k, v, metadata)

        # ── 2. GQA repeat_kv ──
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # 对齐 dtype（MPS bf16 下 cache gather 可能改变 dtype）
        if k.dtype != q.dtype:
            k = k.to(q.dtype)
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        # ── 3. Scaled dot-product attention ──
        attn_weights = torch.matmul(q, k.transpose(2, 3)).mul_(self.scaling)

        # causal mask（仅 seq_len > 1 时构建，decode T=1 跳过）
        if seq_len > 1:
            seq_k = k.shape[-2]
            if self.kv_cache is not None and isinstance(self.kv_cache, LayerKVCache):
                diagonal = cache_position + 1
            else:
                diagonal = 1
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_k, dtype=torch.bool, device=q.device),
                diagonal=diagonal,
            )
            attn_weights = attn_weights.masked_fill(
                causal_mask[None, None, :, :],
                torch.finfo(attn_weights.dtype).min,
            )

        # valid_lens mask（M4 paged 或 M3 batched decode/prefill）
        if paged_valid_lens is not None:
            attn_weights = self._build_valid_lens_mask(attn_weights, paged_valid_lens)
        elif cache_positions is not None:
            if seq_len == 1:
                # M3 decode: cache_positions 是写入位置，valid_lens = cache_positions + 1
                attn_weights = self._build_batched_mask(attn_weights, cache_positions)
            else:
                # M3 batched prefill: cache_positions 是 seq_lens，直接用作 valid_lens
                attn_weights = self._build_valid_lens_mask(attn_weights, cache_positions)

        # softmax（fp32 精度）
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # ── 4. Attention output ──
        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.num_heads * self.head_dim)
        )
        return attn_output

    # ── M2: single cache RW ──
    def _single_cache_rw(self, k, v, metadata):
        """M2 路径：全局 cache_position 写入 + 切片读取。"""
        cache = self.kv_cache
        seq_len = k.shape[-2]
        # 使用 metadata.cur_len（Python int）避免 GPU→CPU 同步
        cur_len = metadata.cur_len if metadata.cur_len is not None else int(metadata.seq_lens[0])
        if seq_len == cur_len:
            cache_position = 0  # prefill: 从头写
        else:
            cache_position = cur_len - 1  # decode: 写最后 1 个位置
        cache.k[:, :, cache_position : cache_position + seq_len, :] = k
        cache.v[:, :, cache_position : cache_position + seq_len, :] = v
        k = cache.k[:, :, : cache_position + seq_len, :]
        v = cache.v[:, :, : cache_position + seq_len, :]
        return k, v, cache_position

    # ── M3: batched cache RW ──
    def _batched_cache_rw(self, k, v, metadata):
        """M3 路径：per-slot 写入 + gather 读取。"""
        cache = self.kv_cache
        seq_lens = metadata.seq_lens.to(k.device)
        is_decode = k.shape[-2] == 1  # T=1 → decode

        if is_decode:
            cache_positions = seq_lens - 1  # [B] 每个 slot 的写入位置
            slot_mapping = metadata.slot_mapping  # [B] tensor
            for i in range(len(seq_lens)):
                # 通过 metadata.slot_mapping 或直接遍历
                # 暂时简化：假设 cache 的 slot 顺序和 batch 顺序一致
                slot = int(slot_mapping[i])
                pos = int(cache_positions[i])
                cache.k[slot, :, pos : pos + 1, :] = k[i]
                cache.v[slot, :, pos : pos + 1, :] = v[i]
            max_len = int(seq_lens.max().item())
            # gather + 清零无效位置
            k = cache.k[slot_mapping, :, :max_len, :]
            v = cache.v[slot_mapping, :, :max_len, :]
            # 清零无效 K/V
            valid_lens = seq_lens.to(k.device)
            positions = torch.arange(max_len, device=k.device)
            valid = positions[None, :] < valid_lens[:, None]
            invalid = ~valid[:, None, :, None]
            k = k.masked_fill(invalid, 0)
            v = v.masked_fill(invalid, 0)
            return k, v, cache_positions
        else:
            # prefill: 写整段到 slot
            slot_mapping = metadata.slot_mapping  # [B] tensor
            for i in range(k.shape[0]):
                slot = int(slot_mapping[i])
                plen = int(seq_lens[i])
                cache.k[slot, :, :plen, :] = k[i, :, :plen, :]  # 只写有效位置
                cache.v[slot, :, :plen, :] = v[i, :, :plen, :]
            k = cache.k[slot_mapping, :, : int(seq_lens.max()), :]
            v = cache.v[slot_mapping, :, : int(seq_lens.max()), :]
            # 返回 seq_lens 触发 valid_lens mask（padded 时屏蔽无效位置）
            return k, v, seq_lens

    # ── M4: paged cache RW ──
    def _paged_cache_rw(self, k, v, metadata):
        """M4 paged: 用 block_table tensor 做 scatter/gather。"""
        cache = self.kv_cache
        layer = cache.layers[self.layer_idx]
        block_table = metadata.block_table.to(k.device)  # [B, max_blocks]
        seq_lens = metadata.seq_lens.to(k.device)  # [B]
        B = k.shape[0]
        is_prefill = k.shape[2] > 1

        # ── Write: scatter k/v → physical blocks ──
        # layer.k: [num_blocks, block_size, n_kv, D]
        num_blocks, block_size, n_kv, D = layer.k.shape
        flat_cache_k = layer.k.view(-1, n_kv, D)
        flat_cache_v = layer.v.view(-1, n_kv, D)

        if is_prefill:
            # prefill: 写入整段 prompt
            for i in range(B):
                slen = int(seq_lens[i])
                for pos in range(slen):
                    block_idx = pos // block_size
                    offset = pos % block_size
                    phys = int(block_table[i, block_idx])
                    flat_cache_k[phys * block_size + offset] = k[i, :, pos, :]
                    flat_cache_v[phys * block_size + offset] = v[i, :, pos, :]
        else:
            # decode: 写入单个 token (pos = seq_lens - 1)
            for i in range(B):
                pos = int(seq_lens[i]) - 1
                block_idx = pos // block_size
                offset = pos % block_size
                phys = int(block_table[i, block_idx])
                flat_cache_k[phys * block_size + offset] = k[i, :, 0, :]
                flat_cache_v[phys * block_size + offset] = v[i, :, 0, :]

        # ── Read: gather from block_table ──
        # block_table [B, nb] → layer.k[block_table] → [B, nb, bs, n_kv, D]
        gathered_k = layer.k[block_table]
        gathered_v = layer.v[block_table]
        # reshape: [B, nb, bs, n_kv, D] → [B, nb*bs, n_kv, D] → [B, n_kv, L, D]
        L = gathered_k.shape[1] * gathered_k.shape[2]
        gathered_k = gathered_k.reshape(B, L, n_kv, D).transpose(1, 2)
        gathered_v = gathered_v.reshape(B, L, n_kv, D).transpose(1, 2)

        # 清零 valid_lens 之外的 K/V，防止 NaN padding 通过 matmul 传播
        # 注意：不能用 * mask，因为 NaN * 0 = NaN（IEEE 754）
        positions = torch.arange(L, device=gathered_k.device)
        valid_mask = positions[None, :] < seq_lens[:, None]  # [B, L]
        gathered_k = gathered_k.masked_fill(~valid_mask[:, None, :, None], 0.0)
        gathered_v = gathered_v.masked_fill(~valid_mask[:, None, :, None], 0.0)

        valid_lens = seq_lens
        return gathered_k, gathered_v, valid_lens

    # ── mask helpers ──
    def _build_valid_lens_mask(self, scores, valid_lens):
        """M4 valid_lens mask。"""
        valid = self._build_valid_positions(scores.shape[-1], valid_lens, scores.device)
        return scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)

    def _build_valid_positions(self, seq_len, valid_lens, device):
        """构建 [B, seq_len] 布尔 mask。"""
        positions = torch.arange(seq_len, device=device)
        valid_lens = valid_lens.to(device)
        return positions[None, :] < valid_lens[:, None]

    def _build_batched_mask(self, scores, cache_positions):
        """M3 per-row mask。"""
        max_len = int(cache_positions.max().item()) + 1
        valid_lens = (cache_positions + 1).to(scores.device)
        positions = torch.arange(max_len, device=scores.device)
        visible = positions[None, :] < valid_lens[:, None]
        return scores.masked_fill(~visible[:, None, None, :], torch.finfo(scores.dtype).min)


# ── 2. Qwen3Attention（上层）──
# 模型层调用的入口：projection → QK-norm → RoPE → Attention → o_proj。
# 对齐 vLLM V1 Qwen3Attention：forward(positions, hidden_states)。
class Qwen3Attention(nn.Module):
    """Qwen3 self-attention：projection + QK-norm + RoPE + Attention + o_proj。

    对齐 vLLM V1 Qwen3Attention.forward(positions, hidden_states)。
    内部委托 Attention 做 cache RW + attention 计算。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads

        # projection 层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK-norm + RoPE
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)

        # 下层 Attention（cache RW + attention 计算）
        self.attn = Attention(config)

    @override
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Qwen3 self-attention forward。

        Args:
            positions: [B, T] 绝对位置（RoPE 用）
            hidden_states: [B, T, H]

        Returns:
            [B, T, H]
        """
        B, T, _ = hidden_states.shape

        # 1. Q/K/V projection + reshape
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_proj(hidden_states)
            .view(B, T, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(B, T, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        # 2. QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. RoPE
        cos, sin = self.rotary_emb(q, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # 4. 委托 Attention 做 cache RW + attention 计算
        attn_output = self.attn(q, k, v)

        # 5. Output projection
        return self.o_proj(attn_output)
