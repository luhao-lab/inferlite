"""KV Cache 数据结构。

## 为什么需要 KV Cache？

Transformer 自回归生成时，每一步 Decode 需要对所有历史 token 做 Attention：

    output_t = softmax(Q_t · K_{0..t}ᵀ / √d) · V_{0..t}

如果不缓存，每一步都要重算所有历史 token 的 K/V，复杂度 O(T²)。
KV Cache 把每一层每个 token 产生的 K/V 存下来，Decode 阶段只算当前新 token
的 Q，用缓存的 K/V 做 Attention，复杂度降到 O(T)。

## 两阶段使用方式

    Prefill 阶段（输入 prompt，一次性处理所有 token）：
        for layer_idx, layer in enumerate(model.layers):
            k, v = layer.attention.project_kv(x)           # [B, n_kv, T_prompt, D]
            cache.layers[layer_idx].k[:, :, :T_prompt, :] = k
            cache.layers[layer_idx].v[:, :, :T_prompt, :] = v
        cache.cur_len = T_prompt

    Decode 阶段（逐 token 生成）：
        while not done:
            pos = cache.cur_len
            k_new, v_new = layer.attention.project_kv(x_new)  # [B, n_kv, 1, D]
            cache.layers[i].k[:, :, pos:pos+1, :] = k_new     # 追加写入
            cache.layers[i].v[:, :, pos:pos+1, :] = v_new
            # 用 k[:, :, :pos+1, :] 和 v[:, :, :pos+1, :] 做 Attention
            cache.cur_len += 1

## 设计要点

1. 静态预分配（Static Allocation）
   from_config 一次性分配 num_hidden_layers 个 [B, n_kv, max_seq_len, D] tensor，
   推理期间不再 malloc，避免内存碎片和 GPU OOM 风险。

2. cur_len 是唯一事实源
   所有层共享一个 cur_len，表示当前有效 token 数。
   Attention 用 k[:, :, :cur_len, :] 切片读取有效部分，不依赖 tensor 内容是否为零。

3. reset() 只清 cur_len，不清零 tensor
   节约时间：下次 Prefill 会从 [:T_prompt] 覆盖写入，旧数据不影响结果。
"""

from dataclasses import dataclass

import torch

from inferlite.config import ModelConfig


@dataclass
class LayerKVCache:
    """单层 Transformer 的 KV Cache 容器。

    存储一个 Decoder Layer 在推理过程中累积的 Key 和 Value 矩阵。
    所有槽位在 from_config 时一次性预分配（静态），推理时做切片写入。

    Attributes:
        k: Key cache，shape [B, n_kv_heads, max_seq_len, head_dim]。
           第三维是预分配的最大序列槽位，有效范围由 KVCache.cur_len 决定。
        v: Value cache，shape 与 k 完全相同。
    """

    k: torch.Tensor  # [B, n_kv_heads, max_seq_len, head_dim]
    v: torch.Tensor  # [B, n_kv_heads, max_seq_len, head_dim]


class KVCache:
    """全模型 KV Cache 容器。

    管理所有 Decoder Layer 的缓存，并维护当前有效 token 数（cur_len）。
    通常由 EngineCore 在一次 generate() 调用开始时创建，生成结束后释放。

    Attributes:
        layers: 每个元素对应一个 Decoder Layer 的缓存，
                len(layers) == config.num_hidden_layers。
        cur_len: 当前已写入的有效 token 数，所有层共享。
                 Prefill 后等于 prompt 长度；每次 Decode 步 +1。
                 这是 Attention 切片的唯一事实源：k[:, :, :cur_len, :]。
    """

    def __init__(self, layers: list[LayerKVCache]) -> None:
        self.layers = layers
        self.cur_len: int = 0  # 唯一事实源，初始为 0

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> "KVCache":
        """按模型配置静态预分配所有层的 KV Cache。

        一次性分配 num_hidden_layers 个 LayerKVCache，每个包含
        shape [batch_size, num_key_value_heads, max_seq_len, head_dim]
        的 k/v tensor，初始值为全零。

        Args:
            config:      模型超参，从中读取层数、KV head 数、head_dim。
            batch_size:  并发请求数，M2 阶段固定为 1。
            max_seq_len: 最长序列槽位，超过会触发 IndexError（越界写入）。
            dtype:       tensor 精度，通常 torch.float32（MPS 兼容）。
            device:      计算设备，"cpu" / "mps" / "cuda"。

        Returns:
            KVCache 实例，cur_len=0，所有 tensor 已分配到指定 device。
        """
        layers = []
        for _ in range(config.num_hidden_layers):
            # k 和 v 的 shape 完全相同，zeros_like 复用 shape/dtype/device
            k = torch.zeros(
                batch_size,
                config.num_key_value_heads,
                max_seq_len,
                config.head_dim,
                dtype=dtype,
                device=device,
            )
            v = torch.zeros_like(k)
            layers.append(LayerKVCache(k=k, v=v))
        return cls(layers)

    def reset(self) -> None:
        """重置 cur_len 为 0，准备接受下一次 Prefill。

        只清 cur_len，不清零 tensor：
        - 下次 Prefill 会从索引 0 开始覆盖写入，旧数据不影响结果。
        - 避免对大 tensor 做不必要的 memset，节约时间。
        """
        self.cur_len = 0
