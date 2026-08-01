# M4-T8 — Attention Backend Refactor + 模型链瘦身

> 在 T7 统一 engine loop + CacheAdapter 后，进一步整理模型链：把 `GQAAttention.forward` 中不断增长的 M1/M2/M3/M4 cache 分支收敛到 `AttentionCacheContext` + `KVAccessResult`，同时把模型链（`Qwen3ForCausalLM` → `Qwen3Model` → `DecoderLayer`）的 12 个透传参数收敛为单个 `cache_ctx`，为 M5 Prefix Cache 和 M9 kernel backend 做准备。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T8 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ⬜ pending |
| 前置 | M4-T7 Engine Loop Unification ✅ |
| 后续 | M4-T9 — Docs & Tag |
| 估时 | 4～5h |
| 核心文件 | `inferlite/model/attention.py`、`inferlite/model/qwen3.py` |
| 新增文件 | `inferlite/model/attention_cache.py` |
| 改造文件 | `inferlite/engine/protocol.py`、`inferlite/cache/adapter.py` |
| 测试文件 | M1-M4 全量既有测试 |
| 产物 | attention 主流程瘦身、模型链参数收敛、cache context/result 抽象 |

## 背景

M4 完成 T5（paged engine 集成）后，模型链已经膨胀到：

```text
GQAAttention.forward:     11 参数，4 路分支（M1/M2/M3/M4），535 行
DecoderLayer.forward:     12 参数（含 layer_idx），纯透传
Qwen3Model.forward:       11 参数，纯透传
Qwen3ForCausalLM.forward: 11 参数，纯透传
LLMModel Protocol:        9 参数
```

每个里程碑加 2-3 个 cache 参数，穿透 4 层。如果 M5 Prefix Cache、M9 kernel backend 继续直接往这里加分支，模型链会变成里程碑堆叠的技术债。

T7 已经用 `CacheAdapter` 统一了 engine loop 的 cache 差异。T8 的任务是让模型链也收敛——cache 参数不再穿透每一层，而是通过单个 `AttentionCacheContext` 传递。

## 范围冻结

T8 是 **结构整理任务**，不是新能力任务。

### 明确做

**Part A：Attention 层瘦身**（原 T7 范围）

- 整理 `GQAAttention.forward` 中 M1/M2/M3/M4 cache 分支。
- 引入 `AttentionCacheContext` + `KVAccessResult`。
- 让 `GQAAttention.forward` 回到稳定主流程：`projection → q/k norm → RoPE → cache/backend → attention core → o_proj`。
- 保留 M4 PyTorch gather 版 PagedAttention 作为教学/fallback 实现。

**Part B：模型链瘦身**（新增范围）

- `DecoderLayer.forward` 签名从 12 参数收敛到 `hidden_states, position_ids, position_embeddings, cache_ctx`。
- `Qwen3Model.forward` 同理收敛；内部 `enumerate(self.layers)` 自动填充 `layer_idx` 到 context。
- `Qwen3ForCausalLM.forward` 同理收敛。
- `LLMModel` Protocol 参数收敛（与 T7 的 adapter 配合）。

### 明确不做

- 不实现 vLLM / nano-vLLM 的 CUDA/Triton/FlashAttention kernel（M9）。
- 不引入 Prefix Cache 命中、hash、LRU、CoW 行为（M5）。
- 不修改 scheduler 语义。
- 不把 request/block metadata tensor 化为生产级 `slot_mapping` / `block_tables` / `seq_lens`（M9）。
- 不为了重构改动 logits/token 语义。
- 不删除 M3 fixed-slot 路径（它仍是 oracle）。

## 核心设计

### 1. AttentionCacheContext

```python
# model/attention_cache.py

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class AttentionCacheContext:
    """统一的 cache 调用上下文，替代模型链中散落的 cache 参数。

    对应 vLLM 的 AttentionMetadata / ForwardContext 教学简化版。
    每个 model forward 调用只需传一个 context，不传 8 个独立参数。
    """
    mode: Literal["none", "single", "batched", "paged"]
    # M2 参数
    layer_kv_cache: object = None  # LayerKVCache | BatchedLayerKVCache
    cache_position: int = 0
    # M3 参数
    cache_slots: object = None     # torch.Tensor
    cache_positions: object = None # torch.Tensor
    # M4 参数
    paged_kv_cache: object = None  # PagedKVCache
    layer_idx: int = 0
    request_ids: list = field(default_factory=list)
    is_prefill: bool = False

    @staticmethod
    def none() -> "AttentionCacheContext":
        """M1 无 cache 路径。"""
        return AttentionCacheContext(mode="none")

    @staticmethod
    def single(layer_kv_cache, cache_position: int) -> "AttentionCacheContext":
        """M2 单序列 cache 路径。"""
        return AttentionCacheContext(mode="single", layer_kv_cache=layer_kv_cache,
                                     cache_position=cache_position)

    @staticmethod
    def batched(layer_kv_cache, cache_slots, cache_positions=None) -> "AttentionCacheContext":
        """M3 fixed-slot batched cache 路径。"""
        return AttentionCacheContext(mode="batched", layer_kv_cache=layer_kv_cache,
                                     cache_slots=cache_slots, cache_positions=cache_positions)

    @staticmethod
    def paged(paged_kv_cache, layer_idx: int, request_ids: list, is_prefill: bool) -> "AttentionCacheContext":
        """M4 paged cache 路径。"""
        return AttentionCacheContext(mode="paged", paged_kv_cache=paged_kv_cache,
                                     layer_idx=layer_idx, request_ids=request_ids,
                                     is_prefill=is_prefill)
```

### 2. KVAccessResult

```python
@dataclass
class KVAccessResult:
    """cache backend 返回的统一结果。

    包含 attention core 消费 K/V 和构造 mask 所需的全部信息。
    """
    k: object   # torch.Tensor — 已包含当前 token/prompt 可见历史的 K
    v: object   # torch.Tensor — 已包含当前 token/prompt 可见历史的 V
    valid_lens: object = None  # torch.Tensor | None — 变长 batch 的 per-row 有效长度
    causal_offset: int = 0     # prefill causal mask 的 query 起始绝对位置
```

### 3. GQAAttention.forward 瘦身

```python
# 改造前（11 参数，4 路分支散落）
def forward(self, hidden_states, position_ids, position_embeddings,
            layer_kv_cache, cache_position, cache_slots, cache_positions,
            paged_kv_cache, layer_idx, request_ids, is_prefill):
    ...
    if paged_kv_cache is not None:
        ...  # 30 行 paged 逻辑
    elif isinstance(layer_kv_cache, BatchedLayerKVCache):
        ...  # 30 行 batched 逻辑
    elif layer_kv_cache is not None:
        ...  # 20 行 single 逻辑
    else:
        ...  # 10 行 no-cache 逻辑
    ...

# 改造后（5 参数，主流程清晰）
def forward(self, hidden_states, position_ids, position_embeddings,
            cache_ctx: AttentionCacheContext | None = None):
    q, k, v = self._project_qkv(hidden_states)
    q, k = self._apply_qk_norm_and_rope(q, k, position_ids, position_embeddings)
    kv_result = self._cache_rw(k, v, cache_ctx)
    out = self._attention_core(q, kv_result)
    return self.o_proj(out)
```

`_cache_rw` 内部根据 `cache_ctx.mode` 分发到 `_single_cache_rw` / `_batched_cache_rw` / `_paged_cache_rw`，但返回统一的 `KVAccessResult`。

### 4. 模型链瘦身

```python
# DecoderLayer.forward 改造前（12 参数）
def forward(self, hidden_states, position_ids, position_embeddings,
            layer_kv_cache, cache_position, cache_slots, cache_positions,
            paged_kv_cache, layer_idx, request_ids, is_prefill):
    hidden_states = self.self_attn(hidden_states, position_ids, position_embeddings,
                                    layer_kv_cache=layer_kv_cache, cache_position=cache_position,
                                    cache_slots=cache_slots, cache_positions=cache_positions,
                                    paged_kv_cache=paged_kv_cache, layer_idx=layer_idx,
                                    request_ids=request_ids, is_prefill=is_prefill)

# DecoderLayer.forward 改造后（4 参数）
def forward(self, hidden_states, position_ids, position_embeddings,
            cache_ctx: AttentionCacheContext | None = None):
    hidden_states = self.self_attn(hidden_states, position_ids, position_embeddings,
                                    cache_ctx=cache_ctx)

# Qwen3Model.forward 改造后：内部自动填充 layer_idx
def forward(self, input_ids, position_ids, position_embeddings,
            cache_ctx: AttentionCacheContext | None = None):
    for idx, layer in enumerate(self.layers):
        # 对 paged 路径，自动填充 layer_idx
        if cache_ctx is not None and cache_ctx.mode == "paged":
            layer_ctx = replace(cache_ctx, layer_idx=idx)
        else:
            layer_ctx = cache_ctx
        hidden_states = layer(hidden_states, position_ids, position_embeddings,
                              cache_ctx=layer_ctx)
```

### 5. 与 T7 CacheAdapter 的衔接

T7 的 `CacheAdapter.prefill_model_kwargs()` / `decode_model_kwargs()` 返回 dict。T8 可以改成返回 `AttentionCacheContext`：

```python
# cache/adapter.py 改造
class CacheAdapter(Protocol):
    def prefill_cache_context(self, request_ids: list[str]) -> AttentionCacheContext: ...
    def decode_cache_context(self, request_ids: list[str]) -> AttentionCacheContext: ...

# engine/loop.py 改造
ctx = cache_adapter.prefill_cache_context(request_ids)
logits = model(batch_input_ids, position_ids=..., cache_ctx=ctx)
```

## 与 vLLM 的关系

| vLLM 概念 | inferlite T8 对应 |
|---|---|
| `ForwardContext` | `AttentionCacheContext` |
| `AttentionMetadata` | `AttentionCacheContext`（合并简化） |
| `Attention wrapper → backend` | `_cache_rw` → `_single/_batched/_paged_cache_rw` |
| kernel output / result | `KVAccessResult` + PyTorch attention core |
| `slot_mapping` | M4 仍由 `PagedKVCache` Python API 内部处理 |
| `block_tables` / `seq_lens` tensor | M9 再导出 |

## 实现步骤

1. 记录当前 M1-M4 测试命令，作为重构前基线。
2. 新建 `model/attention_cache.py`：`AttentionCacheContext` + `KVAccessResult`。
3. `GQAAttention.forward` 内部把旧参数组装成 context（保持外部签名兼容或逐步迁移）。
4. 抽出 `_cache_rw(k, v, cache_ctx)` → 返回 `KVAccessResult`。
5. 抽出 `_attention_core(q, kv_result)` 集中处理 score/mask/softmax/value。
6. 复用 `_build_valid_positions`，避免 M3/M4 mask 重复实现。
7. `DecoderLayer.forward` 签名瘦身，内部传 `cache_ctx`。
8. `Qwen3Model.forward` 签名瘦身，内部 `enumerate` 填 `layer_idx`。
9. `Qwen3ForCausalLM.forward` 签名瘦身。
10. `LLMModel` Protocol 参数收敛。
11. T7 adapter 的 `*_model_kwargs` 改为返回 `AttentionCacheContext`。
12. 跑 M1 回归（`test_attention.py`、`test_qwen3_model.py`）。
13. 跑 M2 回归（`test_attention_kv.py`）。
14. 跑 M3 回归（`test_batched_attention.py`、`test_batch_engine.py`）。
15. 跑 M4 回归（`test_paged_attention.py`、`test_paged_kv_cache.py`、`test_paged_batch_engine.py`）。
16. 更新注释：说明这是 vLLM AttentionMetadata/ForwardContext 思路的教学简化版。
17. 任务卡追加完成总结与 commit 号。

## 测试要求

至少运行：

```bash
uv run ruff check inferlite tests
uv run ruff format --check inferlite tests
uv run pytest tests/ -q
```

M1-M4 全量回归必须全绿。

## DoD

- [ ] `model/attention_cache.py` 存在，包含 `AttentionCacheContext` + `KVAccessResult`。
- [ ] `GQAAttention.forward` 主流程清晰，cache 分支不再散落在 attention 数学逻辑中。
- [ ] `DecoderLayer.forward` / `Qwen3Model.forward` / `Qwen3ForCausalLM.forward` 签名收敛到 `cache_ctx`。
- [ ] M1/M2/M3/M4 行为不变，全量测试通过。
- [ ] M4 PyTorch gather paged attention 仍作为 fallback/oracle 存在。
- [ ] M5 不会继续直接往 `forward` 塞 Prefix Cache 分支。
- [ ] M9 可以基于同一 backend 边界替换为 kernel backend。
- [ ] T7 adapter 与 T8 context 对齐。
- [ ] 任务卡完成总结记录真实命令、结果和 commit。

## 坑（按概率排序）

1. **一次性改外部调用签名**：会扩大影响面。优先保持旧签名，在内部组装 context。
2. **重构时改坏 causal mask 语义**：M2 prefill 的 `cache_position` / paged prefill 的 key 长度要特别小心。
3. **valid_lens 只做 score mask，不清零 K/V**：会重新引入 `0 * NaN = NaN`。
4. **把 Prefix Cache 顺手做进来**：这是 M5，不属于 T8。
5. **把 M9 kernel metadata 提前 tensor 化**：T8 只留边界，不做生产级 metadata。
6. **删除 M3 fixed-slot path**：M3 仍是 oracle，不能删。
7. **helper 参数太多反而更乱**：如果 context 设计不清晰，应先内部 dataclass 化再拆 helper。
8. **模型链瘦身和 attention 瘦身同时做**：建议分两步——先 attention（Part A），再模型链（Part B），每步跑回归。
9. **T7 adapter 返回 dict 和 T8 context 不一致**：T8 最后一步统一 adapter 接口。

## 与 M5 / M9 的衔接

- M5 Prefix Cache 应通过 cache adapter / context 接入，不再污染 `GQAAttention.forward` 或模型链。
- M9 kernel backend 应替换/旁路 M4 PyTorch gather backend，但保持同一上层 attention 主流程。
- T9 Docs/Tag 需要记录 T8 最终形成的 attention/backend 边界和模型链签名。
