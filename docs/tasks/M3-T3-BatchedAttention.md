# M3-T3 Batched Attention

> M3 第三张任务卡：让 decode attention 支持 `cache_slots` / `cache_positions`，同一轮 decode 多请求一起算。

## 元信息
- **任务 ID**: M3-T3
- **里程碑**: M3 — Continuous Batching
- **状态**: ⬜ pending
- **前置**: M3-T2
- **估时**: 4h

## 目标

**要解决什么问题**：

M2 attention decode 只支持单请求 KV Cache：

```text
q: [1, n_heads, 1, head_dim]
kv_cache.cur_len: int
```

M3 需要 batched decode：

```text
q: [B, n_heads, 1, head_dim]
cache_slots: [B]
cache_positions: [B]
```

每个 batch row 对应一个请求，它只能 attend 自己 slot 中的有效 KV，不能跨请求互相 attend。

**做完是什么效果**：

同一轮 decode 可以输入多个 next_token：

```python
next_tokens = torch.tensor([[tok_a], [tok_b], [tok_c]])
cache_slots = torch.tensor([0, 2, 5])
cache_positions = torch.tensor([128, 64, 300])
logits = model(next_tokens, position_ids=cache_positions[:, None], kv_cache=batched_cache, cache_slots=cache_slots)
```

**不做什么（边界）**：

- 不做 prefill batching。
- 不做 flash attention / paged attention kernel。
- 不优化 HBM gather 临时张量。
- 不支持跨请求 attention。
- 不做 sliding window。

## 产出文件

- `inferlite/model/attention.py` 中新增或扩展 batched decode 分支
- `inferlite/model/model.py` / `inferlite/model/layers.py` 必要参数透传
- `tests/unit/test_batched_attention.py`

## 算法核心

### 1. 写入当前 token 的 K/V

当前 token 的 K/V 必须先写进对应 slot 的当前位置：

```python
# k_new/v_new: [B, n_kv_heads, 1, head_dim]
# cache_slots: [B]
# cache_positions: [B]
for i, slot in enumerate(cache_slots.tolist()):
    pos = int(cache_positions[i])
    layer_cache.k[slot, :, pos : pos + 1, :] = k_new[i]
    layer_cache.v[slot, :, pos : pos + 1, :] = v_new[i]
```

### 2. gather 当前 batch 的 K/V

M3 教学版允许 materialize dense KV：

```python
max_len = int(cache_positions.max().item()) + 1
k = layer_cache.k[cache_slots, :, :max_len, :]
v = layer_cache.v[cache_slots, :, :max_len, :]
```

得到：

```text
k/v: [B, n_kv_heads, max_len, head_dim]
```

### 3. per-row mask

每个请求可见长度不同：

```python
valid_lens = cache_positions + 1
positions = torch.arange(max_len, device=device)
visible = positions[None, :] < valid_lens[:, None]

scores = scores.masked_fill(
    ~visible[:, None, None, :],
    torch.finfo(scores.dtype).min,
)
```

### 4. query attend 自己

当前 query 可以 attend 当前 token 自己的 K/V，这是标准 causal self-attention：

```text
q_t attends to k_0 ... k_t
```

不要把当前位置错误 mask 掉。

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|---|---|---|
| 1 | batched decode 输出 shape | `[B, 1, vocab]` 或 hidden `[B,1,D]` | 精确 |
| 2 | cache slot 写入位置正确 | 只写 `slot_i, pos_i` | 精确 |
| 3 | 不同 row 不串 KV | req_a 不读取 req_b KV | 精确 |
| 4 | mask 保留当前位置 | query 能 attend self | 精确 |
| 5 | padding 位置被 mask | masked score 为 dtype min | 精确 |
| 6 | B=1 时和 M2 decode 等价 | logits/hidden close | `atol=1e-4` |
| 7 | 不同 `cache_positions` 混 batch | 结果等价逐条 decode | `atol=1e-4` |
| 8 | MQA/GQA repeat_kv 后 shape 正确 | heads 对齐 | 精确 |

## DoD

- [ ] attention 支持 `BatchedKVCache` decode 分支。
- [ ] model forward 参数能透传 `cache_slots` / `cache_positions`。
- [ ] per-row mask 单测覆盖。
- [ ] B=1 和 M2 decode 等价性测试通过。
- [ ] 多 slot 混合 decode 与逐条 decode 等价性测试通过。
- [ ] `uv run pytest tests/unit/test_batched_attention.py -q` 通过。
- [ ] 现有 M2 generate 测试不回归。
- [ ] commit `feat(attention): support fixed-slot batched decode attention (M3-T3 done)`。

## 坑（按概率排序）

1. **把多个请求的 sequence concat 到一起**：M3 是 batch gather，不是跨请求序列拼接。
2. **mask 误用全局 causal mask**：每 row 的有效长度不同，需要 per-row mask。
3. **漏写当前 token KV 就先 attention**：会导致 query attend 不到自己。
4. **把 `cache_positions` 当成 `seq_lens` 后加一两次**：约定 `cache_positions` 是当前 token 写入位置，visible len 是 `cache_positions + 1`。
5. **破坏 M2 路径**：M2 `KVCache` decode 分支应保持兼容。

## 完成总结

待完成后补：batched decode attention 的接口、mask 语义、和逐条 decode 等价性结果。
