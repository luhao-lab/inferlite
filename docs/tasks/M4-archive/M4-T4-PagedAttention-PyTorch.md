# M4-T4 — PagedAttention PyTorch 伪版

> 把 T3 已完成的 `PagedKVCache` 接入 `GQAAttention`：投影/RoPE 后写入分页 KV，再按 block table gather 成临时连续 KV，用纯 PyTorch 跑出与 fixed-slot 路径等价的 attention 输出。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T4 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ⬜ pending |
| 前置 | M4-T1 BlockPool ✅、M4-T2 BlockTable ✅、M4-T3 PagedKVCache ✅ |
| 后续 | M4-T5 — BatchEngine Integration |
| 估时 | 3～4h |
| 核心文件 | `inferlite/model/attention.py` |
| 测试文件 | `tests/unit/test_paged_attention.py` |
| 参考文档 | `docs/knowledge/m4-paged-kv-cache.md`、`docs/knowledge/lessons.md` L5 |

## 范围冻结

T4 只做 **Attention 层闭环**：直接以 `GQAAttention` 为对象做 unit test，证明 paged KV 路径的数学结果、mask 语义和数值安全正确。

### 明确做

- 让 `GQAAttention.forward` 支持 `PagedKVCache`。
- 使用 T3 的 batch API：`write_prefill` / `write_decode` / `gather_kv`。
- 根据 `valid_lens` 清零 K/V padding，防止 NaN/Inf 传播。
- 根据 `valid_lens` 对 attention scores 做 per-row visible mask。
- 新增 paged attention 单测：B=1 prefill、B=1 decode、B>1 变长 decode、跨 block、NaN padding、参数合同。
- 保持 M1/M2/M3 旧路径不回归。

### 明确不做

- 不打通完整 `Qwen3Model` / `DecoderLayer` / `Qwen3ForCausalLM` 调用链。
- 不接 `engine` / `batch_generate` / scheduler / admission。
- 不在 attention 层调用 `allocate_request` / `append_token` / `free_request`。
- 不做 Prefix Cache、hash、LRU、CoW（留 M5）。
- 不写 Triton / CUDA / MPS kernel（留 M9）。
- 不追求速度优于 M3；T4 只要求机制正确、数值安全和旧路径不回归。

如果实现时发现必须修改 model/engine 调用链，停止并重新规划；不得在 T4 中隐式扩大范围。

## 背景

当前 `GQAAttention.forward` 已有三条路径：

```text
layer_kv_cache is None              -> M1 full causal attention
LayerKVCache                        -> M2 single-sequence KV cache
BatchedLayerKVCache                 -> M3 fixed-slot batch KV cache
```

T3 已经提供分页 KV 容器：

```text
PagedKVCache
├── write_prefill(layer_idx, request_ids, k, v)
├── write_decode(layer_idx, request_ids, k, v)
└── gather_kv(layer_idx, request_ids) -> (k, v, valid_lens)
```

T4 要补上第四条路径：

```text
PagedKVCache + layer_idx + request_ids -> M4 paged attention
```

## API 合同

`GQAAttention.forward` 新增 paged 专用参数，默认值必须保证旧调用不受影响：

```python
paged_kv_cache: PagedKVCache | None = None
layer_idx: int | None = None
request_ids: list[str] | None = None
paged_is_prefill: bool = False
```

入口规则：

- `paged_kv_cache is None`：完全保持现有 M1/M2/M3 行为。
- `paged_kv_cache is not None`：
  - `layer_kv_cache` 必须为 `None`，禁止同一次调用同时传两种 cache。
  - `layer_idx` 必须非空。
  - `request_ids` 必须非空。
  - `len(request_ids) == batch_size`。
- `paged_is_prefill=True`：调用 `write_prefill(layer_idx, request_ids, k, v)`。
- `paged_is_prefill=False`：调用 `write_decode(layer_idx, request_ids, k, v)`；decode 的 token 维应为 1，由 T3 的 shape 合同兜底。

T4 不负责请求生命周期。测试或后续 T5 必须在进入 attention 前完成：

```python
paged_cache.allocate_request(request_id, prompt_len)  # prefill 前
paged_cache.append_token(request_id)                  # decode 写当前 token 前
paged_cache.free_request(request_id)                  # finished 后，T5 负责
```

## 算法核心

### 1. forward 分派顺序

```python
if paged_kv_cache is not None:
    k, v, valid_lens = self._paged_cache_rw(...)
elif layer_kv_cache is not None:
    ...  # 现有 M2/M3 路径
else:
    ...  # M1 无 cache 路径
```

paged 分支应位于旧 cache 分支之前，且入口拒绝 `paged_kv_cache` 与 `layer_kv_cache` 同时非空，避免调试时不清楚实际走哪条路径。

### 2. `_paged_cache_rw`

建议新增 helper：

```python
def _paged_cache_rw(
    self,
    cache: PagedKVCache,
    layer_idx: int,
    request_ids: list[str],
    k: torch.Tensor,
    v: torch.Tensor,
    is_prefill: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
```

职责：

1. 校验 paged 参数合同。
2. prefill 时调用 `cache.write_prefill(layer_idx, request_ids, k, v)`。
3. decode 时调用 `cache.write_decode(layer_idx, request_ids, k, v)`。
4. 调 `cache.gather_kv(layer_idx, request_ids)` 得到 `k, v, valid_lens`。
5. 立即按 `valid_lens` 清零 K/V padding。
6. 返回清零后的 `k/v` 和 `valid_lens`。

### 3. K/V padding 清零

`gather_kv` 返回：

```text
k/v:        [B, n_kv, L_pad, D]
valid_lens: [B]
```

`L_pad = max_num_blocks_in_batch * block_size`，不是 `max(valid_lens)`。padding 区来自 `torch.empty`，可能含 NaN/Inf。

必须在 `repeat_kv` 前执行：

```python
seq_k = k.shape[-2]
positions = torch.arange(seq_k, device=k.device)
valid = positions[None, :] < valid_lens[:, None]
invalid = ~valid[:, None, :, None]
k = k.masked_fill(invalid, 0)
v = v.masked_fill(invalid, 0)
```

只做 score mask 不够。softmax 后 padding 概率即使是 0，value 聚合中仍可能出现：

```text
0 * NaN = NaN
```

这条继承 `docs/knowledge/lessons.md` L5。

### 4. score valid-lens mask

建议新增 helper：

```python
def _build_valid_lens_mask(
    self,
    scores: torch.Tensor,
    valid_lens: torch.Tensor,
) -> torch.Tensor: ...
```

逻辑：

```python
seq_k = scores.shape[-1]
positions = torch.arange(seq_k, device=scores.device)
visible = positions[None, :] < valid_lens[:, None]
scores = scores.masked_fill(~visible[:, None, None, :], torch.finfo(scores.dtype).min)
```

mask 语义：

- causal mask：防止 query 看未来 token。
- valid-lens mask：防止 query 看 block padding。

两者不能相互替代。

### 5. prefill / decode mask 规则

| 场景 | query 长度 | key 长度 | mask |
|---|---:|---:|---|
| paged prefill | `T` | `L_pad` | causal mask + valid-lens mask |
| paged decode | `1` | `L_pad` | valid-lens mask |

prefill 的 causal mask 必须使用 key 维 `k.shape[-2]`，不能假设 key 长度等于输入 `seq_len`。

## 实现步骤

1. 在 `attention.py` 导入 `PagedKVCache` 类型。
2. 扩展 `GQAAttention.forward` 参数，所有新增参数带默认值。
3. 在 q/k/v projection、QK norm、RoPE 后新增 paged 分支。
4. 实现 `_paged_cache_rw`。
5. 实现 `_build_valid_lens_mask`。
6. 在 scores 阶段叠加 paged valid-lens mask。
7. 新增 `tests/unit/test_paged_attention.py`。
8. 跑定向测试和旧路径回归。
9. 补齐实现与测试文件教学级注释。
10. 完成后追加 `## 完成总结` 与 commit 号。

## L0 测试清单

| # | 测什么 | 预期 |
|---|---|---|
| 1 | 旧路径不变 | `test_attention.py`、`test_attention_kv.py`、`test_batched_attention.py` 继续全绿 |
| 2 | B=1 paged prefill | paged 输出与 no-cache full attention 对齐 |
| 3 | B=1 paged decode | prompt 先 prefill，再 append/write decode；decode 输出与 M2 single cache 对齐 |
| 4 | B>1 变长 decode | 不同请求长度合批，paged decode 与 M3 fixed-slot decode 对齐 |
| 5 | 跨 block 边界 | `block_size=4`，长度 4 -> 5 decode 边界输出仍对齐 |
| 6 | padding NaN 不传播 | 物理 cache padding 注入 NaN，paged attention 输出 finite |
| 7 | request_ids 顺序 | 用可区分输入锁定 batch 行与 request id 一一对应 |
| 8 | 参数合同 | 缺 `layer_idx` / 缺 `request_ids` / 同时传 `layer_kv_cache` 与 `paged_kv_cache` 抛错 |

### 测试命令

```bash
uv run pytest tests/unit/test_paged_attention.py -q
uv run pytest tests/unit/test_attention.py tests/unit/test_attention_kv.py tests/unit/test_batched_attention.py -q
uv run pytest tests/unit/test_paged_kv_cache.py -q
```

T4 不改完整 model 调用链，因此默认不要求跑 generate / engine E2E；若实现中意外触及调用链，必须补跑对应测试并重新记录原因。

## DoD

- [ ] `GQAAttention.forward` 支持 paged cache 路径。
- [ ] 旧 M1/M2/M3 attention 路径不回归。
- [ ] paged path 使用 T3 batch API，不退回逐请求逐 token K/V 搬运。
- [ ] K/V padding 按 `valid_lens` 清零，防止 NaN/Inf 传播。
- [ ] scores 按 `valid_lens` 做 per-row mask，padding 不可见。
- [ ] B=1 prefill、B=1 decode、B>1 变长 decode 均有等价测试。
- [ ] 跨 block 边界测试通过。
- [ ] padding NaN 测试通过，输出 finite。
- [ ] 参数合同测试通过。
- [ ] `uv run pytest tests/unit/test_paged_attention.py -q` 全绿。
- [ ] 现有 attention/cache 单测回归全绿。
- [ ] 实现与测试文件补齐教学级注释。
- [ ] 本任务卡追加完成总结与 commit 号。

## 坑（按概率排序）

1. **只做 score mask，忘记清零 K/V**：会重现 M3 的 NaN 传播 bug。
2. **把 `L_pad` 当成 `max(seq_len)`**：mask key 维必须来自 `k.shape[-2]`。
3. **attention 层只拿单层 cache**：T3 的读写 API 是 `PagedKVCache` 级别，必须传完整 cache 和 `layer_idx`。
4. **attention 层偷偷 append_token**：append 属于 T5 生命周期管理，T4 只写当前 K/V。
5. **request_ids 与 batch 行错配**：shape 不报错但语义全错，测试必须锁住顺序。
6. **靠 `seq_len` 猜 prefill/decode**：单 token prompt 与未来 chunked prefill 会出问题，T4 用显式 `paged_is_prefill`。
7. **同时传两种 cache**：必须拒绝，否则很难判断实际走哪条路径。
8. **过早接 model/engine**：T4 是 attention 层任务，完整调用链留 T5。

## 与后续任务的衔接

T5 使用本任务提供的 attention 能力接入 batch engine，并负责：

- request allocate/free；
- decode 前 append_token；
- block admission；
- running / waiting / finished 生命周期。

T6 负责 E2E 与 benchmark。T7 负责 attention/backend 边界整理，T8 负责文档和 tag 收口。
