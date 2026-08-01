# M4-T7 — Attention Backend Refactor

> 在 T4～T6 已经证明 PagedAttention 机制正确后，整理 `GQAAttention.forward` 中不断增长的 M1/M2/M3/M4 分支：把 cache 读写与 mask 元信息收敛到轻量 context/backend 层，让 attention 主流程重新变薄，为 M5 Prefix Cache 和 M9 kernel backend 做准备。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T7 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ⬜ pending |
| 前置 | M4-T6 E2E Correctness & Benchmark ✅ |
| 后续 | M4-T8 — Docs & Tag |
| 估时 | 3～5h |
| 核心文件 | `inferlite/model/attention.py` |
| 可能新增文件 | `inferlite/model/attention_cache.py` 或等价轻量模块 |
| 测试文件 | 既有 M1/M2/M3/M4 attention/cache/batch 测试 |
| 产物 | attention 主流程瘦身、cache context/result 抽象、回归测试记录 |

## 背景

M4-T4 为了最小闭环，把 paged cache 分支直接接进了 `GQAAttention.forward`。这符合 T4 的教学目标，但到 M4 后半段，`forward` 已经同时承担：

```text
qkv projection
q/k norm
RoPE
M1 no-cache
M2 single cache
M3 fixed-slot batched cache
M4 paged cache
causal mask
per-row valid-lens mask
softmax + value 聚合
o_proj
```

如果 M5 Prefix Cache、M9 kernel backend 继续直接往这里加分支，`GQAAttention` 会变成里程碑堆叠的技术债。T7 的职责是在 M4 收口前，把这个边界整理清楚。

## 范围冻结

T7 是 **结构整理任务**，不是新能力任务。

### 明确做

- 整理 `GQAAttention.forward` 中 M1/M2/M3/M4 cache 分支。
- 引入轻量版 `AttentionCacheContext` / `KVAccessResult` 或等价结构。
- 让 `GQAAttention.forward` 回到稳定主流程：

```text
projection -> q/k norm -> RoPE -> cache/backend -> attention core -> o_proj
```

- 保留 M4 PyTorch gather 版 PagedAttention 作为教学/fallback 实现。
- 明确 cache backend 返回哪些 mask 元信息，例如：
  - `valid_lens`：每个 batch row 的有效 KV 长度；
  - `causal_offset` 或等价字段：构造 causal mask 所需的起始位置；
  - 是否需要 per-row valid-length mask。
- 保持 M1/M2/M3/M4 旧路径行为不变。
- 跑既有 attention/cache/batch 回归测试。
- 更新必要注释，说明这是 vLLM AttentionMetadata/backend 思路的教学简化版。

### 明确不做

- 不实现 vLLM / nano-vLLM 的 CUDA/Triton/FlashAttention kernel。
- 不引入 Prefix Cache 命中、hash、LRU、CoW 行为。
- 不修改 scheduler 语义。
- 不把 request/block metadata tensor 化为生产级 `slot_mapping` / `block_tables` / `seq_lens`。
- 不为了重构改动 logits/token 语义。
- 不删除 M3 fixed-slot 路径；它仍是 M4/M5 的 oracle。

## 设计目标

### 1. `GQAAttention.forward` 变薄

目标形态：

```python
q, k, v = self._project_qkv(hidden_states)
q, k = self._apply_qk_norm_and_rope(q, k, position_ids, position_embeddings)
kv = self._cache_rw(k, v, cache_ctx)
out = self._attention_core(q, kv, seq_len)
return self.o_proj(out)
```

不要求一次拆成完全独立 backend 类，但至少要把 cache 读写分支从 attention 数学主流程中隔离出来。

### 2. 统一 cache 结果

建议引入：

```python
@dataclass
class KVAccessResult:
    k: torch.Tensor
    v: torch.Tensor
    valid_lens: torch.Tensor | None = None
    causal_offset: int = 0
```

含义：

- `k/v`：已经包含当前 token/prompt 可见历史的 KV。
- `valid_lens`：如 M3/M4 变长 batch 需要 per-row mask，则返回有效长度。
- `causal_offset`：构造 prefill causal mask 时 query 起始绝对位置；M2 single cache 可等价为 `cache_position`。

字段可按实际实现调整，但必须保证「cache 后端产出 mask 所需元信息，attention core 消费元信息」。

### 3. 统一 cache context

建议引入：

```python
@dataclass
class AttentionCacheContext:
    mode: Literal["none", "single", "batched", "paged"]
    layer_kv_cache: LayerKVCache | BatchedLayerKVCache | None = None
    paged_kv_cache: PagedKVCache | None = None
    cache_position: int = 0
    cache_slots: torch.Tensor | None = None
    cache_positions: torch.Tensor | None = None
    layer_idx: int | None = None
    request_ids: list[str] | None = None
    is_prefill: bool = False
```

如果担心一次性改动过大，可以先在 `forward` 内部把旧参数组装成 context，外部调用签名暂时不变：

```python
cache_ctx = AttentionCacheContext.from_legacy_args(...)
```

这样 T7 不会强迫 T5/T6 同时大改调用链。

### 4. 保留 PyTorch fallback

T7 后仍然是：

```text
M4 paged backend = write_prefill/write_decode + gather_kv + valid_lens mask + matmul attention
```

它不是生产实现，只是 M9 kernel backend 的 correctness oracle。

## 与 vLLM 的关系

vLLM 的模型 attention 层很薄，复杂性主要下沉到：

```text
Attention wrapper -> AttentionMetadata -> backend/kernel
```

T7 不照搬完整 vLLM，只做教学版对应关系：

| vLLM 概念 | inferlite T7 对应 |
|---|---|
| Attention wrapper/backend | `_cache_rw` / 轻量 backend helper |
| AttentionMetadata | `AttentionCacheContext` |
| backend result / kernel output | `KVAccessResult` + PyTorch attention core |
| `slot_mapping` | M4 仍由 `PagedKVCache` Python API 内部处理 |
| `block_tables` / `seq_lens` tensor | M9 再导出 |
| paged attention kernel | M9；T7 不做 |

## 实现步骤

1. 记录当前 T4～T6 通过的测试命令，作为重构前基线。
2. 新增或内嵌 `AttentionCacheContext` / `KVAccessResult`。
3. 把 `GQAAttention.forward` 旧参数转换为 context，保持外部签名兼容。
4. 抽出 `_cache_rw(k, v, cache_ctx, batch_size, seq_len)`：
   - no-cache：原样返回；
   - single cache：调用 `_single_cache_rw`；
   - batched fixed-slot：调用 `_batched_cache_rw` / `_batched_prefill_rw`；
   - paged：调用 `_paged_cache_rw`。
5. 抽出 `_attention_core(q, k, v, seq_len, kv_result)` 或等价 helper，集中处理：
   - `repeat_kv`；
   - score matmul；
   - causal mask；
   - valid-lens mask；
   - softmax；
   - value 聚合。
6. 复用 `_build_valid_positions`，避免 M3/M4 mask 重复实现。
7. 更新 docstring 和注释：明确 M1/M2/M3/M4 支持路径与 backend 思路。
8. 跑回归测试。
9. 任务卡追加完成总结与 commit 号。

## 测试要求

至少运行：

```bash
uv run ruff check inferlite tests
uv run ruff format --check inferlite tests
uv run pytest tests/unit/test_attention.py -q
uv run pytest tests/unit/test_attention_kv.py -q
uv run pytest tests/unit/test_batched_attention.py -q
uv run pytest tests/unit/test_paged_attention.py -q
uv run pytest tests/unit/test_paged_kv_cache.py -q
```

如果某些 paged attention 测试尚未存在，记录原因，并至少保证已存在的 M1/M2/M3/M4 相关测试通过。

## DoD

- [ ] `GQAAttention.forward` 主流程清晰，cache 分支不再散落在 attention 数学逻辑中。
- [ ] 有统一的 cache context/result 或等价结构。
- [ ] M1/M2/M3/M4 行为不变。
- [ ] M4 PyTorch gather paged attention 仍作为 fallback/oracle 存在。
- [ ] M5 以后不会继续直接往 `forward` 塞 Prefix Cache 分支。
- [ ] M9 可以基于同一 backend 边界替换为 kernel backend。
- [ ] ruff / format / attention 相关测试通过或限制记录清楚。
- [ ] 任务卡完成总结记录真实命令、结果和 commit。

## 坑（按概率排序）

1. **一次性改外部调用签名**：会扩大影响面。优先保持旧签名，在内部组装 context。
2. **重构时改坏 causal mask 语义**：M2 prefill 的 `cache_position` / paged prefill 的 key 长度要特别小心。
3. **valid_lens 只做 score mask，不清零 K/V**：会重新引入 `0 * NaN = NaN`。
4. **把 Prefix Cache 顺手做进来**：这是 M5，不属于 T7。
5. **把 M9 kernel metadata 提前 tensor 化**：T7 只留边界，不做生产级 metadata。
6. **删除 M3 fixed-slot path**：M3 仍是 oracle，不能删。
7. **helper 参数太多反而更乱**：如果 context 设计不清晰，应先内部 dataclass 化再拆 helper。

## 与 M5 / M9 的衔接

- M5 Prefix Cache 应通过 cache manager/backend/context 接入，不再污染 `GQAAttention.forward` 主流程。
- M9 kernel backend 应替换/旁路 M4 PyTorch gather backend，但保持同一上层 attention 主流程。
- T8 Docs/Tag 需要记录 T7 最终形成的 attention/backend 边界。
