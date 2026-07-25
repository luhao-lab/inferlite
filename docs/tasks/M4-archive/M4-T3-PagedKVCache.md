# M4-T3 — PagedKVCache

> **状态**：⬜ pending
> **里程碑**：M4 PagedAttention
> **目标**：实现多层分页 KV Cache 容器，支持按 block table 写入和 gather 读取。
> **前置**：M4-T1 BlockPool + M4-T2 BlockTable

## 背景

M3 `BatchedKVCache` 的单层 shape 是 `[max_num_slots, n_kv_heads, max_seq_len, head_dim]`。M4 改为 `[num_blocks, block_size, n_kv_heads, head_dim]`。

## 产出

文件：`inferlite/cache/paged_kv_cache.py`

```python
@dataclass
class PagedLayerKVCache:
    """单层 paged KV 数据容器。
    k/v shape: [num_blocks, block_size, n_kv_heads, head_dim]
    """
    k: torch.Tensor
    v: torch.Tensor


class PagedKVCache:
    """分页 KV Cache。"""

    @classmethod
    def from_config(cls, config, num_blocks, block_size, dtype, device) -> "PagedKVCache": ...

    # ── 请求生命周期 ──
    def allocate_request(self, request_id: str, prompt_len: int) -> None: ...
    def free_request(self, request_id: str) -> None: ...
    def may_append_block(self, request_id: str) -> None: ...

    # ── KV 读写 ──
    def write_prefill(self, layer_idx, request_id, k, v) -> None: ...
    def write_decode(self, layer_idx, request_id, k, v) -> None: ...

    # ── Attention 读取（gather 伪版）──
    def gather_kv(self, layer_idx, request_id) -> tuple[torch.Tensor, torch.Tensor]: ...
```

## 接口语义

### `from_config(config, num_blocks, block_size, dtype, device)`

- 创建 BlockPool + 每层 K/V tensor。
- K/V shape: `[num_blocks, block_size, n_kv_heads, head_dim]`。

### `allocate_request(request_id, prompt_len)`

- 按 `ceil(prompt_len / block_size)` 分配 block。
- 创建 BlockTable 并注册。

### `free_request(request_id)`

- 释放请求的所有 block（逐个调用 `block_pool.free`）。

### `may_append_block(request_id)`

- decode 追加 token 时调用。
- 如果 `needs_new_block()`，先分配新 block 并 append 到 block_table。
- `seq_len += 1`。

### `write_prefill(layer_idx, request_id, k, v)`

- k/v shape: `[n_kv_heads, prompt_len, head_dim]`。
- 按 position_to_block 写入对应 physical block 的 offset。

### `write_decode(layer_idx, request_id, k, v)`

- k/v shape: `[n_kv_heads, 1, head_dim]`。
- 写入 `seq_len - 1` 位置对应的 block offset。

### `gather_kv(layer_idx, request_id) -> tuple[Tensor, Tensor]`

- 按 block_table 逐 block 切片并 cat。
- 返回 k/v shape: `[n_kv_heads, seq_len, head_dim]`。

## 算法核心

- prefill 前按 `ceil(prompt_len / block_size)` 分配 block。
- decode 追加时如果 `seq_len % block_size == 0`，先分配新 block。
- gather 通过 block table 把非连续 block 拼成临时连续 KV。

## 测试

建议新建 `tests/unit/test_paged_kv_cache.py`。

- 单请求 prefill 写入跨多个 block。
- decode 追加跨 block 边界。
- gather_kv 与连续 KV 对齐。
- free_request 释放所有 block。
- 多请求 block table 独立。

## DoD

- [ ] PagedKVCache 单测全过。
- [ ] 不修改 M3 `BatchedKVCache`。
- [ ] 支持 CPU/MPS。
