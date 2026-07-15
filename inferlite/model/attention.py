"""Qwen3 GQA Attention 的最小数值对齐实现。

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

T4 只做 full causal attention，不做 KV cache；KV cache 留到 M2。
"""

from typing import override

import torch
import torch.nn as nn

from inferlite.config import ModelConfig
from inferlite.model import BatchedLayerKVCache
from inferlite.model.kv_cache import LayerKVCache
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


class GQAAttention(nn.Module):
    """Qwen3 decoder self-attention: QK-norm + RoPE + GQA。

    这层只实现“单段 prefill/full attention”：输入一整段 token，输出同长度 hidden states。
    暂不处理 KV cache、sliding window、attention dropout、返回 attention weights 等工程功能。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        # 这些结构超参会在 forward 的 reshape/repeat/scaling 中反复使用，必须挂到 self。
        # 注意：head_dim 直接来自 config.head_dim。Qwen3-0.6B 中 H/n_q=64，但 head_dim=128，
        # 不能用 hidden_size // num_heads 代替。
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.num_key_value_heads: int = config.num_key_value_heads
        self.head_dim: int = config.head_dim
        self.num_key_value_groups: int = self.num_heads // self.num_key_value_heads
        self.scaling: float = self.head_dim**-0.5
        self.rms_norm_eps: float = config.rms_norm_eps
        self.rope_theta: float = config.rope_theta

        # Qwen3 attention projection 都没有 bias。
        # q_proj 输出 n_q * D；k/v_proj 输出 n_kv * D。
        # GQA 减少的是 K/V head 数，不减少 Query head 数。
        self.q_proj: nn.Linear = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.k_proj: nn.Linear = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj: nn.Linear = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=False,
        )
        # o_proj 把拼回来的所有 Query heads 输出重新映射回 hidden_size，供 residual add 使用。
        self.o_proj: nn.Linear = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )

        # Qwen3 在 q/k 做 RoPE 前分别做 RMSNorm，归一化维度是 head_dim。
        # 这里的 eps 必须来自 config.rms_norm_eps，保证和 transformers/Qwen3Config 对齐。
        self.q_norm: RMSNorm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.k_norm: RMSNorm = RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.rotary_emb: RotaryEmbedding = RotaryEmbedding(self.head_dim, self.rope_theta)

    # @override 是给类型检查器看的声明：forward 是在重写 nn.Module.forward。
    # 它不改变运行逻辑，但能防止把 forward 误拼成 foward 这类错误。
    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        layer_kv_cache: LayerKVCache | BatchedLayerKVCache | None = None,
        cache_position: int = 0,
        cache_slots: torch.Tensor | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行 causal self-attention，支持 M1（无 cache）和 M2（有 cache）两路。

        ## 两种调用方式

        **M1 路径**（无 cache，每步传全量 tokens）：
            forward(hidden_states=[B, T, H], position_ids=[B, T])
            - hidden_states 包含历史 + 当前所有 token
            - 每步重算所有 K/V，O(T²)

        **M2 路径**（有 cache，每步只传当前 token）：
            forward(hidden_states=[B, 1, H], position_embeddings=(cos, sin),
                    layer_kv_cache=cache.layers[i], cache_position=cur_len)
            - hidden_states 只含当前新 token（T=1 for decode，T=T_p for prefill）
            - 历史 K/V 从 cache 读取，O(T)

        ## 参数

        Args:
            hidden_states: [B, T, hidden_size]
                M1：所有 token（历史 + 当前）的 hidden states。
                M2 prefill：prompt 全部 token，T = prompt 长度。
                M2 decode：只有当前新 token，T = 1。
            position_ids: [B, T]，M1 兼容路径专用。
                token 的绝对位置，用于 Attention 内部计算 RoPE cos/sin。
                M2 路径改用 position_embeddings（外部统一计算，避免 28 层重复调用）。
            position_embeddings: (cos, sin)，M2 路径专用，与 position_ids 二选一。
                由 Qwen3Model.forward 统一计算后传入；shape: [B, T, head_dim]。
            layer_kv_cache: 当前层的 KV Cache 容器。
                LayerKVCache 时走 M2 路径（单序列 cache）。
                BatchedLayerKVCache 时走 M3 路径（多请求 batched cache）。
                None 时走 M1 兼容路径，行为与原始实现完全相同。
            cache_position: M2 路径专用。当前 token(s) 写入 cache 的起始槽位 = kv_cache.cur_len。
                prefill 时为 0；decode 第 i 步时为 prompt_len + i。
            cache_slots: M3 路径专用。[B] 每个请求对应的 slot id。
            cache_positions: M3 路径专用。[B] 每个请求当前 token 的写入位置。

        Returns:
            [B, T, hidden_size]，T 与输入 hidden_states 的 seq_len 相同。
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. q/k/v projection，并把最后一维拆成 heads × head_dim。
        # projection 后的中间形状：
        #   q: [B, T, n_q * D]
        #   k: [B, T, n_kv * D]
        #   v: [B, T, n_kv * D]
        # view 后变成：
        #   q: [B, T, n_q, D]
        #   k/v: [B, T, n_kv, D]
        q = self.q_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        k = self.k_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        v = self.v_proj(hidden_states).view(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )

        # [B, T, heads, D] -> [B, heads, T, D]。
        # 这样最后两维就是 [T, D]，方便后面做：
        #   q @ k.transpose(-2, -1) => [T, D] @ [D, T] = [T, T]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 2. Qwen3 特有：RoPE 前做 q/k RMSNorm。
        # RMSNorm 的 normalized_shape 是 head_dim，所以它会在每个 head 的 D 维上归一化。
        # 注意：v 不做 RMSNorm；QK-norm 只服务 q/k 相似度计算稳定性。
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 3. RoPE：根据位置生成 cos/sin，旋转 q/k 嵌入位置信息。v 不做 RoPE。
        # M1 路径：在 Attention 内部调用 rotary_emb（每层各算一次，共 28 次）。
        # M2 路径：cos/sin 由 Qwen3Model 统一算好传入，28 层共用一次结果，节省重复计算。
        if position_embeddings is not None:
            cos, sin = position_embeddings  # M2 路径
        else:
            cos, sin = self.rotary_emb(q, position_ids)  # M1 兼容路径
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)

        # 3.5 KV Cache 读写（M2 路径，RoPE 之后 / repeat_kv 之前）。
        #
        # 时机选择：RoPE 之后——k 已经携带位置信息，存入 cache 的是"带位置的 k"。
        # repeat_kv 之前——cache 存的是原始 n_kv 维度，更省显存；repeat 之后再做 attention。
        #
        # 写入：把当前 token(s) 的 k/v 原地写入 cache 的 [cache_position] 槽位。
        #   切片赋值（不是 append/cat），原地修改 cache tensor，不分配新内存。
        # 读出：用切片 [:cache_position + seq_len] 一次拿出所有有效历史。
        #   这一刀切出来的是 cache tensor 的 view（不 copy），T_k = cache_position + seq_len。
        #
        # M1 兼容路径（layer_kv_cache=None）：跳过，k/v 直接用当前输入的。
        if layer_kv_cache is not None:
            if isinstance(layer_kv_cache, BatchedLayerKVCache):
                assert cache_slots is not None
                if cache_positions is not None:
                    k, v = self._batched_cache_rw(
                        layer_kv_cache, k, v, cache_slots, cache_positions
                    )
                else:
                    k, v = self._batched_prefill_rw(layer_kv_cache, k, v, cache_slots, seq_len)
            else:
                k, v = self._single_cache_rw(layer_kv_cache, k, v, cache_position, seq_len)

        # 4. GQA repeat_kv：把 KV heads 扩展到 Query heads 数量，才能一一对应做 attention。
        # cache 存的是 n_kv 维度；repeat 之后 k/v 变成 n_q 维度。
        # 有 cache 时 k/v 的 T 维已经是完整历史 T_k（>= seq_len）。
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # 5. scaled dot-product attention。
        # q:           [B, n_q, T, D]    T = 当前输入长度
        # k（有cache）: [B, n_q, T_k, D]  T_k = cache_position + T >= T
        # attn_weights: q @ k^T → [B, n_q, T, T_k]
        # 第 i 行：第 i 个 query token 对所有 T_k 个历史 token 的打分。
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        # causal mask：防止当前 token 看到未来位置。
        # 只在 seq_len > 1 时构建（prefill 需要；decode 步 T=1 单 token 无未来可看，跳过）。
        # 有 cache 时 k 的 T 维是 T_k（历史 + 当前），mask shape 必须是 [T, T_k] 而非 [T, T]。
        # diagonal = cache_position + 1：prefill（cache_position=0）等价于标准 diagonal=1。
        if seq_len > 1:
            seq_k = k.shape[-2]  # 带 cache 时 T_k > T
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_k, dtype=torch.bool, device=hidden_states.device),
                diagonal=cache_position + 1,
            )
            # 显式扩到 [1, 1, T, T]，广播到 [B, n_q, T, T]。
            causal_mask = causal_mask[None, None, :, :]
            # 被 mask 的位置填成 dtype 最小值，softmax 后概率接近 0。
            attn_weights = attn_weights.masked_fill(
                causal_mask,
                torch.finfo(attn_weights.dtype).min,
            )
        # M3 per-row mask（batched decode，每行可见长度不同）
        # prefill 阶段 cache_positions=None，走标准 causal mask，不需 per-row mask
        if isinstance(layer_kv_cache, BatchedLayerKVCache) and cache_positions is not None:
            attn_weights = self._build_batched_mask(attn_weights, cache_positions)

        # 与 transformers eager attention 对齐：softmax 用 fp32 做，再 cast 回 q dtype。
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # 6. 用 attention 概率聚合 value，再回到 [B, T, hidden_size]。
        # [B, n_q, T, T] @ [B, n_q, T, D] -> [B, n_q, T, D]
        attn_output = torch.matmul(attn_weights, v)
        # [B, n_q, T, D] -> [B, T, n_q, D] -> [B, T, n_q * D]
        # contiguous 是为了保证 transpose 后内存布局可被 view 安全解释。
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.num_heads * self.head_dim,
            )
        )
        # o_proj 回到 hidden_size，输出将进入 DecoderLayer 的 residual add。
        return self.o_proj(attn_output)

    def _single_cache_rw(
        self,
        cache: LayerKVCache,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_position: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """M2 路径：全局 cache_position 写入 + 切片读取。

        写：当前 token(s) 的 k/v → cache[cache_position : cache_position + seq_len]
        读：完整有效历史 k/v → cache[:cache_position + seq_len]（view，零拷贝）。
        """
        cache.k[:, :, cache_position : cache_position + seq_len, :] = k
        cache.v[:, :, cache_position : cache_position + seq_len, :] = v
        k = cache.k[:, :, : cache_position + seq_len, :]
        v = cache.v[:, :, : cache_position + seq_len, :]
        return k, v

    def _batched_cache_rw(
        self,
        cache: BatchedLayerKVCache,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_slots: torch.Tensor,
        cache_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """M3 路径：per-slot 写入 + gather 读取。

        写：每个请求的当前 token k/v 写入自己 slot 的 position（for 循环，B 很小）。
        读：从各 slot gather 到 [B, H_kv, max_len, D]，短请求的尾部是垃圾数据（需 per-row mask）。
        """
        for i, slot in enumerate(cache_slots.tolist()):
            pos = int(cache_positions[i])
            cache.k[slot, :, pos : pos + 1, :] = k[i]
            cache.v[slot, :, pos : pos + 1, :] = v[i]

        max_len = int(cache_positions.max().item()) + 1
        k = cache.k[cache_slots, :, :max_len, :]
        v = cache.v[cache_slots, :, :max_len, :]
        return k, v

    def _batched_prefill_rw(
        self,
        cache: BatchedLayerKVCache,
        k: torch.Tensor,
        v: torch.Tensor,
        cache_slots: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """M3 prefill 路径：B=1，将整个 prompt 的 KV 写入单个 slot。

        与 _batched_cache_rw（decode）的区别：
          - decode: 每 slot 写 1 个 token，需 cache_positions 定位
          - prefill: 单 slot 从位置 0 连续写 seq_len 个 token，不需 cache_positions

        为什么不复用 _single_cache_rw？
          - _single_cache_rw 服务 M2 的 KVCache（第 0 维是 batch，固定 1）
          - _batched_prefill_rw 服务 M3 的 BatchedKVCache（第 0 维是 slot，多请求复用）
          - 两者 shape 都是 [?, H, L, D]，但第 0 维索引方式不同：M2 用 [:]，M3 用 [slot]
          - 类型不同（LayerKVCache vs BatchedLayerKVCache），硬合并需要改签名，不如分开清晰
        """
        slot = int(cache_slots[0])
        cache.k[slot, :, :seq_len, :] = k[0]
        cache.v[slot, :, :seq_len, :] = v[0]
        k = cache.k[slot : slot + 1, :, :seq_len, :]
        v = cache.v[slot : slot + 1, :, :seq_len, :]
        return k, v

    def _build_batched_mask(
        self,
        scores: torch.Tensor,
        cache_positions: torch.Tensor,
    ) -> torch.Tensor:
        """M3 per-row visible mask：每行只能 attend 到自己的有效 KV。

        cache_positions[i] + 1 = 请求 i 的有效 KV 长度。
        超出部分填 dtype min，softmax 后概率 ≈ 0。
        """
        max_len = int(cache_positions.max().item()) + 1
        valid_lens = cache_positions + 1  # [B]
        positions = torch.arange(max_len, device=scores.device)  # [max_len]
        visible = positions[None, :] < valid_lens[:, None]  # [B, max_len]
        # 扩到 [B, 1, 1, max_len] 广播到 [B, n_heads, T, max_len]
        scores = scores.masked_fill(~visible[:, None, None, :], torch.finfo(scores.dtype).min)
        return scores
