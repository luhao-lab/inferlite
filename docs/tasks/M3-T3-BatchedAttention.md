# M3-T3 Batched Attention

> M3 第三张任务卡：让 decode attention 支持 `cache_slots` / `cache_positions`，同一轮 decode 多请求一起算。

## 元信息
- **任务 ID**: M3-T3
- **里程碑**: M3 — Continuous Batching
- **状态**: ✅ done
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

- `inferlite/model/attention.py` — 扩展 forward + 3 个私有方法
- `inferlite/model/qwen3.py` — DecoderLayer / Qwen3Model forward 参数透传
- `tests/unit/test_batched_attention.py`

## 设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 扩展 vs 新建类 | 扩展现有 GQAAttention | q_proj/k_proj/o_proj 等权重共享，不需要重复 |
| cache 逻辑组织 | 私有方法抽取 | 保持 forward 主流程可读 |
| M3 prefill 路径 | 复用 `_batched_cache_rw`（B=1） | M3 cache 第一维是 slot，不能走 M2 的 `_single_cache_rw` |
| per-row mask | `_build_batched_mask` 独立方法 | 与 M2 causal mask 逻辑不同，分开清晰 |

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

## 实现步骤

### Step 1: attention.py — 私有方法 + forward 改造

1. 新增 `_single_cache_rw(cache, k, v, cache_position, seq_len)` — 提取现有 M2 cache 逻辑
2. 新增 `_batched_cache_rw(cache, k, v, cache_slots, cache_positions)` — per-slot 写入 + gather
3. 新增 `_build_batched_mask(cache_positions, max_len, device, dtype)` — per-row mask
4. 修改 `forward()` 签名：加 `cache_slots`, `cache_positions` 参数
5. 修改 `forward()` 内部：cache 分支用 `isinstance` 分派；mask 分支加 per-row mask
6. 注意 import `BatchedLayerKVCache`

### Step 2: qwen3.py — 参数透传

1. `DecoderLayer.forward()` 加 `cache_slots`, `cache_positions`，传给 `self.self_attn`
2. `Qwen3Model.forward()` 加 `cache_slots`, `cache_positions`，传给每层 `DecoderLayer`
3. 注意 M2 generate loop 不传这些参数（默认 None），不受影响

### Step 3: 测试

1. 先写 cache slot 写入位置测试（L0-2）
2. 写不同 row 不串 KV 测试（L0-3）
3. 写 per-row mask 测试（L0-4, L0-5）
4. 写 B=1 与 M2 decode 等价性测试（L0-6）
5. 写多 slot 混合 decode 等价性测试（L0-7）
6. 跑全量回归确保 M2 不受影响

## 坑（按概率排序）

1. **把多个请求的 sequence concat 到一起**：M3 是 batch gather，不是跨请求序列拼接。
2. **mask 误用全局 causal mask**：每 row 的有效长度不同，需要 per-row mask。
3. **漏写当前 token KV 就先 attention**：会导致 query attend 不到自己。
4. **把 `cache_positions` 当成 `seq_lens` 后加一两次**：约定 `cache_positions` 是当前 token 写入位置，visible len 是 `cache_positions + 1`。
5. **破坏 M2 路径**：M2 `KVCache` decode 分支应保持兼容。

## 完成总结

### 接口

```python
# attention.py — GQAAttention.forward 新增参数
def forward(self, ..., layer_kv_cache, cache_position=0,
            cache_slots=None, cache_positions=None)

# qwen3.py — DecoderLayer/Qwen3Model forward 透传
def forward(self, ..., cache_slots=None, cache_positions=None)
```

### mask 语义

- M2 causal mask：`seq_len > 1` 时构建，防止看到未来
- M3 per-row mask：`isinstance(BatchedLayerKVCache)` 时构建，每行只看自己有效 KV

### 等价性验证

- B=1 batched decode ≈ M2 single decode（atol=1e-4）
- 混合 batch decode ≈ 逐条 sequential decode（atol=1e-4）
- 165/165 全量回归通过，M2 路径不受影响
