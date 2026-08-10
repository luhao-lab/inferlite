# M4-T7 — vLLM V1 Architecture Alignment

> 对齐 vLLM V1 的核心架构模式：引入 `ForwardContext` context manager + `CacheAdapter` Protocol + `bind_kv_cache`，统一 engine loop，拆分 `GQAAttention` 为 `Qwen3Attention` + `Attention`，瘦身模型链到 `(input_ids, positions)` 2 参数调用。消除 `batch_core.py` 和 `paged_core.py` 80% 的重复代码，同时把 `GQAAttention.forward` 的 11 参数 / 4 路分叉收敛为 `Attention.forward(q, k, v)` 3 参数 / 1 条主路径。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M4-T7 |
| 里程碑 | M4 — PagedAttention |
| 状态 | ✅ done |
| 前置 | M4-T6 E2E Correctness & Benchmark ✅ |
| 后续 | M4-T8 Docs & Tag |
| 估时 | 5～6h |
| 核心文件 | `engine/forward_context.py`（新建）、`cache/adapter.py`（新建，5 个 adapter）、`engine/loop.py`（新建） |
| 改造文件 | `engine/batch_core.py`、`engine/paged_core.py`、`engine/core.py`、`engine/protocol.py`、`model/qwen3.py`、`model/attention.py`、`cache/paged_kv_cache.py` |
| 测试文件 | 既有 M1-M4 全量测试 |
| 产物 | ForwardContext + CacheAdapter + 统一 engine loop + 瘦身模型链 |

## 背景

M4-T5/T6 完成后，系统存在两个架构问题：

### 问题 1：Engine 层重复

`batch_core.py`（196 行）和 `paged_core.py`（264 行）主循环 80% 重复，只有 cache 操作不同。

### 问题 2：模型链参数膨胀

```
GQAAttention.forward:     11 参数，4 路 if/elif 分叉（M1/M2/M3/M4）
DecoderLayer.forward:     12 参数，纯透传
Qwen3Model.forward:       8 参数，isinstance 分叉
Qwen3ForCausalLM.forward: 9 参数，纯透传
LLMModel Protocol:        9 参数
```

### 与 vLLM V1 的 diff（不必要的）

| 维度 | vLLM V1 | inferlite 现状 | 必要性 |
|---|---|---|---|
| model forward | `(input_ids, positions)` — 2 参数 | 9 参数 | 不必要 |
| cache 传递 | `bind_kv_cache` + ForwardContext | 穿透 4 层 forward | 不必要 |
| attention 结构 | `Qwen3Attention` + `Attention` 分离 | `GQAAttention` 合并 | 不必要 |
| attention 路径 | 1 条主路径 + pluggable backend | 4 路 if/elif 硬编码 | 不必要 |
| RoPE 计算 | per-layer（对齐 vLLM） | paged 路径每层重算 | 不必要 |
| prefill 策略 | 统一 batched | M3 逐条，M4 batched | 不必要 |

## 范围冻结

T7 是 **架构整理任务**，不是新能力任务。

### 明确做

**Part A：ForwardContext + CacheAdapter（engine 层）**

- 新建 `engine/forward_context.py`：`ForwardContext` + `AttentionMetadata` + `set_forward_context()` context manager + `get_forward_context()`。
- 新建 `cache/adapter.py`：`CacheAdapter` Protocol + 5 个 adapter：
  - `NoCacheAdapter`（M1 无 cache）
  - `SingleCacheAdapter`（M2 单序列 KVCache）
  - `BatchedCacheAdapter`（M3 batched BatchedKVCache）
  - `PagedCacheAdapter`（M4 paged PagedKVCache）
- 新建 `engine/loop.py`：公共 `batch_generate_loop()`，从 `paged_core.py` 抽取。
- 改造 `batch_core.py`：薄包装（< 50 行）。
- 改造 `paged_core.py`：薄包装（< 50 行）。
- 改造 `engine/core.py`：`generate()` 适配 ForwardContext + adapter。
- 改造 `engine/protocol.py`：LLMModel Protocol 瘦身。
- 统一 batched prefill：M3 也改为 batched prefill（不再逐条）。

**Part B：模型链瘦身（model 层）**

- 拆分 `GQAAttention` → `Qwen3Attention`（QKV+RoPE）+ `Attention`（独立 attention 层）。
- `Attention.forward(query, key, value)` — 3 参数，cache 通过 `bind_kv_cache` 绑定，metadata 通过 `get_forward_context()` 获取。
- `Qwen3Attention.forward(positions, hidden_states)` — 对齐 vLLM V1 签名。
- `Qwen3DecoderLayer.forward(positions, hidden_states)` — 对齐 vLLM V1 签名和参数顺序。
- `Qwen3Model.forward(input_ids, positions)` — 2 参数，去掉 isinstance 分叉。
- `Qwen3ForCausalLM.forward(input_ids, positions)` — 2 参数。
- `LLMModel` Protocol 瘦身：`(input_ids, *, positions)` 2 参数。
- `bind_kv_cache`：engine loop 初始化时将 cache 绑定到每个 Attention 层。

### 明确不做

- 不实现 vLLM 的 CUDA/Triton/FlashAttention kernel（M9）。
- 不引入 Prefix Cache 命中、hash、LRU、CoW 行为（M5）。
- 不修改 scheduler 内部逻辑。
- 不为了重构改动 logits/token 语义。
- 不删除 M2 `generate()` 单请求入口（它仍是简单场景的便捷路径）。
- 不做 CUDA Graph / cudagraph runtime。

## 核心设计

### 1. ForwardContext + AttentionMetadata（对齐 vLLM V1）

```python
# engine/forward_context.py

from contextlib import contextmanager
from dataclasses import dataclass
import torch

@dataclass
class AttentionMetadata:
    """per-forward 的 attention 元数据。纯 tensor，不含 cache 对象引用。

    对齐 vLLM V1 的 FlashAttentionMetadata。
    cache 对象通过 bind_kv_cache 绑定在 Attention 层上，不经过 metadata 传递。
    """
    num_seqs: int                           # batch 中的请求数
    seq_lens: torch.Tensor                  # [num_seqs] 每个请求的序列长度
    slot_mapping: torch.Tensor | None = None  # [num_tokens] M3 batched 路径的 slot 映射
    block_table: torch.Tensor | None = None   # [num_seqs, max_blocks] M4 paged 路径的 block 表

@dataclass
class ForwardContext:
    """全局前向上下文。对齐 vLLM V1 的 ForwardContext。"""
    attn_metadata: AttentionMetadata

_forward_context: ForwardContext | None = None

def get_forward_context() -> ForwardContext:
    assert _forward_context is not None, "ForwardContext not set"
    return _forward_context

@contextmanager
def set_forward_context(attn_metadata: AttentionMetadata):
    global _forward_context
    _forward_context = ForwardContext(attn_metadata)
    try:
        yield
    finally:
        _forward_context = None
```

### 2. Attention 层 + bind_kv_cache（对齐 vLLM V1）

```python
# model/attention.py

class Attention(nn.Module):
    """独立的 attention 层。对齐 vLLM V1 的 Attention 类。

    KV cache 通过 bind_kv_cache 绑定为 self.kv_cache，
    attention metadata 通过 get_forward_context().attn_metadata 获取。
    forward 只接收 (query, key, value)，不传任何 cache 参数。
    """
    def __init__(self, num_heads, num_kv_heads, head_dim, ...):
        self.kv_cache = None  # 占位，init 时绑定

    def bind_kv_cache(self, kv_cache):
        """初始化时由 engine loop 调用，绑定 cache 到 attention 层。"""
        self.kv_cache = kv_cache

    def forward(self, query, key, value):
        """对齐 vLLM V1：forward 只接收 q, k, v。

        KV cache 写入：根据 self.kv_cache 类型分发到具体读写逻辑。
        Attention metadata：从 get_forward_context().attn_metadata 获取。
        """
        metadata = get_forward_context().attn_metadata
        # 写 cache + 读完整历史 K/V → 返回 attention 输出
        ...
```

### 3. 模型链（对齐 vLLM V1 命名和签名）

```python
# model/qwen3.py

class Qwen3ForCausalLM(nn.Module):
    def forward(self, input_ids, positions):
        """对齐 vLLM V1：(input_ids, positions) 2 参数。
        cache 信息不出现在签名里。
        """
        hidden_states = self.model(input_ids, positions)
        return hidden_states  # logits 由 model runner 单独计算

class Qwen3Model(nn.Module):
    def forward(self, input_ids, positions):
        """对齐 vLLM V1 Qwen3Model.forward。"""
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)

class Qwen3DecoderLayer(nn.Module):
    def forward(self, positions, hidden_states):
        """对齐 vLLM V1 Qwen3DecoderLayer.forward。
        参数顺序：positions 在前，hidden_states 在后。
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states
        # MLP...
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Qwen3Attention(nn.Module):
    """对齐 vLLM V1 Qwen3Attention：QKV projection + RoPE + 调用独立 Attention 层。

    与旧 GQAAttention 的区别：
    - QKV projection 和 attention 分离为两层
    - RoPE 在 Attention 外部计算（per-layer，对齐 vLLM）
    - Attention 层通过 bind_kv_cache 绑定 cache，forward 只接收 (q, k, v)
    """
    def __init__(self, ...):
        self.q_proj = ...
        self.k_proj = ...
        self.v_proj = ...
        self.q_norm = ...
        self.k_norm = ...
        self.rotary_emb = ...
        self.attn = Attention(...)  # 独立的 Attention 层
        self.o_proj = ...

    def forward(self, positions, hidden_states):
        """对齐 vLLM V1 Qwen3Attention.forward。"""
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        # RoPE
        q, k = self.rotary_emb(positions, q, k)
        # Attention（cache 操作在 Attention.forward 内部）
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)
```

### 4. CacheAdapter + bind_kv_cache（对齐 vLLM KVCacheManager）

```python
# cache/adapter.py

class CacheAdapter(Protocol):
    """Engine loop 与 cache 实现之间的适配层。
    对齐 vLLM V1 的 KVCacheManager 接口（简化版）。
    decode 时序（append → 读 position → +1）由各 adapter 在 make_decode_metadata 内部处理。
    """
    # ── 生命周期 ──
    def can_admit(self, prompt_len: int) -> bool: ...
    def allocate(self, request_id: str, prompt_len: int) -> None: ...
    def free(self, request_id: str) -> None: ...

    # ── 元数据构造（decode 时序封装在 make_decode_metadata 内部）──
    def make_prefill_metadata(self, requests) -> AttentionMetadata: ...
    def make_decode_metadata(self, requests) -> AttentionMetadata: ...

    # ── cache 绑定 ──
    def bind_kv_cache(self, model) -> None:
        """初始化时将 cache 绑定到模型的每个 Attention 层。
        对齐 vLLM V1 的 kv_cache 直接赋值模式。
        """
        ...

class NoCacheAdapter:
    """M1 无 cache 的 adapter。所有 cache 操作为空，只构造 metadata。"""
    def can_admit(self, prompt_len: int) -> bool: return True
    def allocate(self, request_id: str, prompt_len: int) -> None: pass
    def free(self, request_id: str) -> None: pass
    def bind_kv_cache(self, model):
        for layer in model.model.layers:
            layer.self_attn.attn.kv_cache = None  # M1 无 cache

class SingleCacheAdapter:
    """M2 单序列 KVCache 的 adapter。"""
    def __init__(self, cache: KVCache): ...
    def can_admit(self, prompt_len: int) -> bool: return True
    def bind_kv_cache(self, model):
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.attn.kv_cache = self.cache.layers[i]

class BatchedCacheAdapter:
    """M3 fixed-slot BatchedKVCache 的 adapter。"""
    def __init__(self, cache: BatchedKVCache): ...
    def bind_kv_cache(self, model):
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.attn.kv_cache = self.cache.layers[i]

class PagedCacheAdapter:
    """M4 paged PagedKVCache 的 adapter。"""
    def __init__(self, cache: PagedKVCache): ...
    def bind_kv_cache(self, model):
        for i, layer in enumerate(model.model.layers):
            layer.self_attn.attn.kv_cache = self.cache
            layer.self_attn.attn.layer_idx = i
```

### 5. Engine Loop（对齐 vLLM execute_model）

```python
# engine/loop.py

def batch_generate_loop(model, sampler, scheduler, adapter, prompts,
                        max_new_tokens, eos_token_id=None, device="cpu",
                        metrics=None):
    """公共 batch generation 主循环。对齐 vLLM V1 execute_model 模式。

    初始化时调用 adapter.bind_kv_cache(model) 绑定 cache 到 Attention 层。
    每次 forward 前用 set_forward_context(metadata) 设置元数据。
    model forward 只接收 (input_ids, positions)。
    decode 时序（append → 读 position → +1）封装在各 adapter 的 make_decode_metadata 内部。
    """
    # ── 初始化：绑定 cache 到模型 ──
    adapter.bind_kv_cache(model)

    while scheduler.has_unfinished():
        # ── 1. Admit + Allocate ──
        admitted = _admit(scheduler, adapter)
        for req in admitted:
            adapter.allocate(req.request_id, req.prompt_ids.shape[1])

        # ── 2. Batched Prefill ──
        if admitted:
            input_ids, positions = _build_prefill_batch(admitted, device)
            metadata = adapter.make_prefill_metadata(admitted)
            with set_forward_context(metadata):
                hidden_states = model(input_ids, positions=positions)
            logits = model.compute_logits(hidden_states)
            _sample_prefill(logits, admitted, sampler, metadata, metrics)

        # ── 3. Decode（时序由 make_decode_metadata 内部处理）──
        running = list(scheduler.running.values())
        if not running:
            break
        metadata = adapter.make_decode_metadata(running)
        next_tokens, positions = _build_decode_batch(running, device)
        with set_forward_context(metadata):
            hidden_states = model(next_tokens, positions=positions)
        logits = model.compute_logits(hidden_states)
        _sample_decode(logits, running, sampler, adapter, metrics)
```

### 6. LLMModel Protocol 瘦身

```python
class LLMModel(Protocol):
    """最小 LLM 推理协议（对齐 vLLM V1）。"""
    def __call__(self, input_ids, *, positions=None) -> torch.Tensor: ...
    def compute_logits(self, hidden_states) -> torch.Tensor: ...
```

## 与 vLLM V1 的对应关系

| vLLM V1 | inferlite T7 | 对齐程度 |
|---|---|---|
| `ForwardContext` | `ForwardContext` | 同名，字段精简 |
| `FlashAttentionMetadata` | `AttentionMetadata` | 纯 tensor，无 cache 对象 |
| `set_forward_context()` | `set_forward_context()` | 同名 context manager |
| `get_forward_context()` | `get_forward_context()` | 同名 |
| `Attention.kv_cache` 直接赋值 | `adapter.bind_kv_cache()` 设置 `attn.kv_cache` | 同效果，inferlite 用显式方法 |
| `Attention.forward(q, k, v)` | `Attention.forward(q, k, v)` | 同名，3 参数 |
| `Qwen3Attention.forward(positions, hidden_states)` | `Qwen3Attention.forward(positions, hidden_states)` | 同名同签名 |
| `Qwen3DecoderLayer.forward(positions, hidden_states)` | `Qwen3DecoderLayer.forward(positions, hidden_states)` | 同名同签名 |
| `Qwen3ForCausalLM.forward(input_ids, positions)` | `Qwen3ForCausalLM.forward(input_ids, positions)` | 同名同签名 |
| `Qwen3ForCausalLM.compute_logits(hidden_states)` | `Qwen3ForCausalLM.compute_logits(hidden_states)` | 同名，logits 与 forward 分离 |
| `GPUModelRunner.execute_model()` | `batch_generate_loop()` | 结构对齐 |
| `V1KVCacheManager` | `PagedCacheAdapter` | 接口对齐 |
| `V0KVCacheManager` | `BatchedCacheAdapter` | 接口对齐 |

## 实现步骤

**Phase 1：Engine 层（Part A）**

1. 记录基线：`uv run pytest tests/ -q`（270 passed）。
2. 新建 `engine/forward_context.py`：`ForwardContext` + `AttentionMetadata` + context manager。
3. 新建 `cache/adapter.py`：`CacheAdapter` Protocol + 5 个 adapter（NoCache / Single / Batched / Paged）。
4. 新建 `engine/loop.py`：从 `paged_core.py` 抽取公共主循环 + `bind_kv_cache` 调用。
5. 改造 `batch_core.py`：薄包装（统一 batched prefill）。
6. 改造 `paged_core.py`：薄包装。
7. 改造 `engine/protocol.py`：LLMModel Protocol 瘦身 + `compute_logits`。
8. 改造 `engine/core.py`：`generate()` 适配 ForwardContext + SingleCacheAdapter / NoCacheAdapter。
9. 跑 M1 测试（`test_attention.py`、`test_qwen3_model.py`）。
10. 跑 M2 测试（`test_attention_kv.py`、`test_generate_kv.py`）。
11. 跑 M3 batch 测试（`test_batch_engine.py`）。
12. 跑 M4 paged 测试（`test_paged_batch_engine.py` + `test_paged_batch_generate.py`）。

**Phase 2：Model 层（Part B）**

9. 新建 `Attention` 类：独立 attention 层，`bind_kv_cache` + `forward(q, k, v)`。
10. 拆分 `GQAAttention` → `Qwen3Attention`（QKV+RoPE）+ `Attention`。
11. `Qwen3DecoderLayer.forward` 签名对齐：`(positions, hidden_states)`。
12. `Qwen3Model.forward` 签名对齐：`(input_ids, positions)`。
13. `Qwen3ForCausalLM.forward` 签名对齐：`(input_ids, positions)`。
14. `LLMModel` Protocol 瘦身。
15. adapter 实现 `bind_kv_cache`：初始化时绑定 cache 到 Attention 层。
16. 跑 M1 回归（`test_attention.py`、`test_qwen3_model.py`）。
17. 跑 M2 回归（`test_attention_kv.py`）。
18. 全量回归（270 tests）。

**Phase 3：收口**

19. 更新注释：说明这是 vLLM V1 ForwardContext 架构的教学简化版。
20. 任务卡追加完成总结与 commit 号。

## 测试要求

至少运行：

```bash
uv run pytest tests/ -q
```

M1-M4 全量回归必须全绿（270 tests）。

## DoD

- [ ] `engine/forward_context.py` 存在，包含 `ForwardContext` + `AttentionMetadata`（纯 tensor）+ context manager。
- [ ] `cache/adapter.py` 存在，包含 `CacheAdapter` Protocol + 5 个 adapter（NoCache / Single / Batched / Paged）。
- [ ] `engine/loop.py` 存在，包含公共 `batch_generate_loop()`（含 `bind_kv_cache` 调用）。
- [ ] `batch_core.py` 瘦身为薄包装（< 50 行）。
- [ ] `paged_core.py` 瘦身为薄包装（< 50 行）。
- [ ] `engine/core.py` 的 `generate()` 适配 ForwardContext + NoCacheAdapter / SingleCacheAdapter。
- [ ] `Attention` 类存在：`bind_kv_cache` + `forward(query, key, value)` 3 参数，4 种 cache 模式内部分发。
- [ ] `Qwen3Attention` 拆分完成：QKV+RoPE 与 Attention 分离。
- [ ] `Qwen3DecoderLayer.forward(positions, hidden_states)` 对齐 vLLM V1 签名。
- [ ] `Qwen3ForCausalLM.forward(input_ids, positions)` 2 参数 + `compute_logits`。
- [ ] `LLMModel` Protocol 瘦身到 2 参数 + `compute_logits`。
- [ ] M1（无 cache）/ M2（单序列 cache）/ M3（batched）/ M4（paged）行为不变，全量测试通过。
- [ ] `PagedKVCache` 接口对齐 vLLM：`write` / `gather` 消费 `block_table` tensor，不再需要 `request_ids`。
- [ ] M5 只需新增一个 `PrefixCacheAdapter` 即可接入 engine loop。
- [ ] 任务卡完成总结记录真实命令、结果和 commit。

## 坑（按概率排序）

1. **GQAAttention 拆分影响面大**：拆成 `Qwen3Attention` + `Attention` 后，M1-M4 所有引用 `GQAAttention` 的测试和模型初始化都要改。建议先新建类，再逐步替换。
2. **`bind_kv_cache` 时机**：必须在第一次 model forward 之前完成。loop 初始化时调用，确保每个 Attention 层的 `self.kv_cache` 已绑定。
3. **Attention.forward 签名变化**：从 `(hidden_states, ...)` 变成 `(query, key, value)`。QKV projection 移到 `Qwen3Attention`，原来在 `GQAAttention.forward` 里做的 QKV 拆分要上移。
4. **M1/M2 测试用 FakeModel 不兼容新 Protocol**：LLMModel Protocol 瘦身后，FakeModel 的 `__call__` 签名也要对齐。逐个检查修改。
5. **DecoderLayer 返回签名变化**：vLLM 返回 `(hidden_states, residual)` tuple，我们不实现 fused norm 可以只返回 `hidden_states`。但要确保 Qwen3Model 的循环调用方式正确。
6. **Batched prefill 改动影响 M3 测试**：M3 原来逐条 prefill，改成 batched 后 `_batched_prefill_rw` 路径可能被首次触发。确保 batched prefill 的 attention mask 正确。
7. **M2 `generate()` 单请求入口**：`core.py` 的 `generate()` 直接调 model，不走 ForwardContext。需要适配或保持独立路径。
8. **一次性改太多**：Phase 1 先改 engine 层（adapter + loop），跑通后再改 Phase 2 模型层。不要同时改。
9. **RoPE 计算位置变化**：vLLM 在每个 `Qwen3Attention.forward` 里调 `self.rotary_emb(positions, q, k)`，是 per-layer 计算。我们之前是 Model 层统一算一次。T7 对齐 vLLM per-layer 计算，牺牲少量效率换架构一致性。
10. **PagedKVCache 接口改造**：`write_prefill` / `write_decode` / `gather_kv` 都要从 `request_ids` 参数改成消费 `block_table` tensor。adapter 负责把 `request_ids` → tensor 的转换吃掉，attention 层只看到 tensor。改完要确保 M4 所有测试（`test_paged_attention.py`、`test_paged_batch_engine.py`、`test_paged_batch_generate.py`）全绿。

## M1-M3 适配设计

> T7 不仅改 M4 代码，还要让 M1（无 cache）/ M2（单序列 cache）/ M3（batched cache）的路径都走新框架。

### M1 适配（无 cache 路径）

**现状**：`GQAAttention.forward(hidden_states, position_ids)` → 内部自算 RoPE → 无 cache 读写 → attention core。

**T7 适配**：
1. `GQAAttention` → 拆为 `Qwen3Attention` + `Attention`。
2. `Attention.kv_cache = None` 时，跳过 cache 读写，直接用当前 q/k/v 做 attention。
3. M1 仍需 `ForwardContext`：`AttentionMetadata(num_seqs=1, seq_lens=[T], slot_mapping=None, block_table=None)`。
4. 新增 `NoCacheAdapter`（轻量）：
   - `bind_kv_cache(model)` → 每层 `attn.bind_kv_cache(None)`
   - `make_prefill_metadata` / `make_decode_metadata` → 构造无 cache 的 `AttentionMetadata`
   - 无 `allocate`/`free`/`prepare_decode`/`commit_decode`（都不需要）

**改动文件**：`cache/adapter.py`（新增 `NoCacheAdapter`，~30 行）

### M2 适配（单序列 cache 路径）

**现状**：`core.py::generate()` 直接调 `engine.model(input_ids, position_ids=pos, kv_cache=cache)`，不经过 ForwardContext。

**T7 适配**：
1. 新增 `SingleCacheAdapter`（包装 M2 的 `KVCache`）：
   - `bind_kv_cache(model)` → 每层 `attn.bind_kv_cache(kv_cache.layers[i])`
   - `make_prefill_metadata` → `AttentionMetadata(num_seqs=1, seq_lens=[T_p])`
   - `make_decode_metadata` → `AttentionMetadata(num_seqs=1, seq_lens=[cur_len])`
2. `core.py::generate()` 改造：
   - 创建 `SingleCacheAdapter` → `adapter.bind_kv_cache(model.model)`
   - M1 路径：用 `NoCacheAdapter` + `set_forward_context`
   - M2 路径：用 `SingleCacheAdapter` + `set_forward_context`
   - model 调用统一为 `model(input_ids, positions=positions)`
   - logits 通过 `model.compute_logits(hidden_states)` 获取
3. `Attention._single_cache_rw` 保留，从 `GQAAttention` 移到 `Attention` 内部。

**改动文件**：
- `cache/adapter.py`（新增 `SingleCacheAdapter`，~50 行）
- `engine/core.py`（改造 `generate()`，~30 行改动）

### M3 适配（batched cache 路径）

**现状**：`batch_core.py` 196 行独立主循环，直接调 `model(..., cache_slots=..., cache_positions=...)`。

**T7 适配**：
1. `batch_core.py` 瘦身为 ~30 行薄包装：创建 `BatchedCacheAdapter` + scheduler → 调 `batch_generate_loop()`。
2. `BatchedCacheAdapter` 已在伪代码中设计完成。
3. M3 改为 **batched prefill**（原来逐条 prefill），与 M4 统一。
4. `Attention._batched_cache_rw` + `_batched_prefill_rw` 保留，移到 `Attention` 内部。

**改动文件**：
- `engine/batch_core.py`（196 → ~30 行）
- `cache/adapter.py`（`BatchedCacheAdapter`，已在伪代码中）

### 模型层统一适配

**`model/attention.py` 改动总览**：

```python
class Attention(nn.Module):
    """独立 attention 层。M1/M2/M3/M4 四种 cache 模式统一在此。"""

    def __init__(self, num_heads, num_kv_heads, head_dim, scaling):
        self.kv_cache = None  # bind_kv_cache 后赋值
        self.layer_idx = None

    def bind_kv_cache(self, kv_cache, layer_idx=None):
        self.kv_cache = kv_cache
        self.layer_idx = layer_idx

    def forward(self, query, key, value):
        metadata = get_forward_context().attn_metadata
        # reshape q/k/v → [B, heads, T, D]
        # dispatch by self.kv_cache type:
        #   None              → M1 路径（无 cache，直接用 q/k/v）
        #   LayerKVCache      → M2 路径（_single_cache_rw）
        #   BatchedLayerKVCache → M3 路径（_batched_cache_rw / _batched_prefill_rw）
        #   PagedKVCache      → M4 路径（_paged_cache_rw）
        # repeat_kv + attention core + output reshape

    def _single_cache_rw(self, ...): ...   # 从 GQAAttention 移入
    def _batched_cache_rw(self, ...): ...   # 从 GQAAttention 移入
    def _batched_prefill_rw(self, ...): ... # 从 GQAAttention 移入
    def _paged_cache_rw(self, ...): ...     # 从 GQAAttention 移入

class Qwen3Attention(nn.Module):
    """QKV projection + QK norm + RoPE + 调用独立 Attention 层。"""
    def __init__(self, config):
        self.q_proj = ...; self.k_proj = ...; self.v_proj = ...
        self.q_norm = ...; self.k_norm = ...
        self.rotary_emb = ...
        self.attn = Attention(...)  # 独立 attention 层
        self.o_proj = ...

    def forward(self, positions, hidden_states):
        q/k/v projection → reshape → QK norm → RoPE → self.attn(q, k, v) → o_proj
```

**`model/qwen3.py` 改动总览**：

| 类 | 现状 | T7 后 |
|---|---|---|
| `DecoderLayer.self_attn` | `GQAAttention(config)` | `Qwen3Attention(config)` |
| `DecoderLayer.forward` | 12 参数 | `(positions, hidden_states)` 2 参数 |
| `Qwen3Model.forward` | 8 参数 + isinstance 分叉 | `(input_ids, positions)` 2 参数 |
| `Qwen3Model.rotary_emb` | Model 层统一算一次 | **删除**（per-layer 在 Qwen3Attention 里算） |
| `Qwen3ForCausalLM.forward` | 9 参数 | `(input_ids, positions)` 2 参数 |
| `Qwen3ForCausalLM.compute_logits` | 不存在 | **新增**：`lm_head(hidden_states)` |

### Adapter 全景

```
cache/adapter.py
├── CacheAdapter (Protocol)        # 公共接口
├── NoCacheAdapter                 # M1 无 cache（新增）
├── SingleCacheAdapter             # M2 单序列 KVCache（新增）
├── BatchedCacheAdapter            # M3 batched BatchedKVCache
└── PagedCacheAdapter              # M4 paged PagedKVCache
```

每个 adapter 都实现：
- `can_admit` / `allocate` / `free`（M1 为空操作）
- `prepare_decode` / `decode_positions` / `commit_decode`
- `make_prefill_metadata` / `make_decode_metadata` → `AttentionMetadata`
- `bind_kv_cache(model)` → 绑定 cache 到每层 `Attention`

### PagedKVCache 接口对齐（消除 request_ids）

**现状**：`_paged_cache_rw` 需要 `request_ids: list[str]` 来查找 `BlockTable`，attention 层知道请求身份。

**T7 对齐 vLLM**：attention 层只看 tensor，不知道 request_id。

```
改造前（request_ids 贯穿）：
    cache.write_prefill(layer_idx, request_ids, k, v)
    k, v, valid_lens = cache.gather_kv(layer_idx, request_ids)

改造后（block_table tensor 贯穿）：
    metadata = get_forward_context().attn_metadata
    cache.write(layer_idx, metadata, k, v)
    k, v, valid_lens = cache.gather(layer_idx, metadata.block_table, metadata.seq_lens)
```

**`PagedKVCache` 需要改的接口**：

| 旧接口 | 新接口 | 说明 |
|---|---|---|
| `write_prefill(layer_idx, request_ids, k, v)` | `write(layer_idx, metadata, k, v)` | metadata 含写入位置信息 |
| `write_decode(layer_idx, request_ids, k, v)` | 同上，`write` 统一 | prefill/decode 由 metadata 区分 |
| `gather_kv(layer_idx, request_ids)` | `gather(layer_idx, block_table, seq_lens)` | 直接用 tensor 索引物理 block |

**`PagedCacheAdapter.make_*_metadata` 负责**：
- 从 `PagedKVCache.block_tables` 提取 `block_table: [num_seqs, max_blocks]` tensor
- 从 `PagedKVCache.block_tables[rid].seq_len` 提取 `seq_lens: [num_seqs]` tensor
- 写入位置（prefill 偏移 / decode 位置）由 adapter 计算并编码到 metadata

**效果**：`Attention.forward` 和 `_paged_cache_rw` 完全不知道 `request_ids` 的存在，只看 tensor。

### `engine/core.py` 改造后

```python
class EngineCore:
    def step(self, input_ids, positions=None):
        """M1 单步：无 cache，positions 自动生成。"""
        if positions is None:
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        metadata = AttentionMetadata(num_seqs=1, seq_lens=torch.tensor([input_ids.shape[1]]))
        with set_forward_context(metadata):
            hidden_states = self.model(input_ids, positions=positions)
        logits = self.model.compute_logits(hidden_states)
        return self.sampler(logits[:, -1, :])

def generate(engine, input_ids, max_new_tokens, eos_token_id=None, kv_cache=None):
    if kv_cache is None:
        # M1 路径：NoCacheAdapter
        adapter = NoCacheAdapter()
    else:
        # M2 路径：SingleCacheAdapter
        adapter = SingleCacheAdapter(kv_cache)

    adapter.bind_kv_cache(engine.model.model)

    if kv_cache is not None:
        kv_cache.reset()
        # Prefill
        T_p = input_ids.shape[1]
        positions = torch.arange(T_p, device=input_ids.device).unsqueeze(0)
        metadata = adapter.make_prefill_metadata(T_p)
        with set_forward_context(metadata):
            hidden_states = engine.model(input_ids, positions=positions)
        logits = engine.model.compute_logits(hidden_states)
        next_token = engine.sampler(logits[:, -1, :])
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # Decode loop
        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            adapter.prepare_decode("0")
            pos_val = adapter.decode_positions(["0"])[0]
            positions = torch.tensor([[pos_val]], device=input_ids.device)
            metadata = adapter.make_decode_metadata(["0"])
            with set_forward_context(metadata):
                hidden_states = engine.model(next_token, positions=positions)
            logits = engine.model.compute_logits(hidden_states)
            adapter.commit_decode(["0"])
            next_token = engine.sampler(logits[:, -1, :])
            input_ids = torch.cat([input_ids, next_token], dim=1)
    else:
        # M1 路径：每步 full forward
        for _ in range(max_new_tokens):
            next_token = engine.step(input_ids)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return input_ids
```

### `cli.py` 改动

```python
# 改动前：
output_ids = generate(engine, input_ids, ..., kv_cache=kv_cache)

# 改动后：不变！generate() 内部已适配。
# cli.py 只需确保 generate() 调用签名不变。
```

### 改动量汇总

| 文件 | 改动类型 | 估计行数 |
|---|---|---|
| `model/attention.py` | 拆分 GQAAttention → Qwen3Attention + Attention | ~300 行重写 |
| `model/qwen3.py` | 改签名 + 删 rotary_emb + 加 compute_logits | ~100 行改 |
| `cache/adapter.py` | 新建 5 个 adapter | ~350 行新 |
| `engine/forward_context.py` | 新建 | ~35 行新 |
| `engine/loop.py` | 新建（从 paged_core 抽取） | ~120 行新 |
| `engine/batch_core.py` | 196 → 30 行 | 删代码 |
| `engine/paged_core.py` | 264 → 30 行 | 删代码 |
| `engine/protocol.py` | 9 参数 → 2 参数 | ~10 行改 |
| `engine/core.py` | generate() 适配 ForwardContext | ~40 行改 |
| `cli.py` | 几乎不变 | ~5 行改 |
| **总计** | | **~1000 行** |

## 与 M5 / M9 的衔接

- **M5 Prefix Cache**：只需新增 `PrefixCacheAdapter`（包装带 hash/LRU 的 PagedKVCache），engine loop 和模型链不变。
- **M9 kernel backend**：`Attention.forward` 内部的 cache 读写替换为 Triton/FlashAttention kernel，上层 attention 主流程不变。
- **M10 Chunked Prefill**：adapter 的 `can_admit` 可以检查 chunked 分配，loop 不变。

## 伪代码框架实现（完整端到端流程）

> 以下伪代码展示 T7 完成后的完整调用链，用于评估设计合理性。

### A. 入口：batch_core.py / paged_core.py（各 ~30 行）

```python
# engine/batch_core.py — M3 入口（薄包装）
def batch_generate(model, sampler, prompts, max_new_tokens,
                   max_num_slots, config, max_seq_len, ...):
    cache = BatchedKVCache.from_config(config, max_num_slots, max_seq_len, ...)
    adapter = BatchedCacheAdapter(cache, max_num_slots)
    scheduler = FCFSScheduler(max_num_seqs=max_num_slots)
    for i, prompt in enumerate(prompts):
        scheduler.submit(RequestState(request_id=str(i), prompt_ids=prompt, ...))
    return batch_generate_loop(model, sampler, scheduler, adapter, prompts,
                                max_new_tokens, ...)

# engine/paged_core.py — M4 入口（薄包装）
def batch_generate_paged(model, sampler, prompts, max_new_tokens,
                         num_blocks, block_size, config, ...):
    cache = PagedKVCache.from_config(config, num_blocks, block_size, ...)
    adapter = PagedCacheAdapter(cache)
    scheduler = FCFSScheduler(max_num_seqs=num_blocks)
    for i, prompt in enumerate(prompts):
        scheduler.submit(RequestState(request_id=str(i), prompt_ids=prompt, ...))
    return batch_generate_loop(model, sampler, scheduler, adapter, prompts,
                                max_new_tokens, ...)
```

### B. 公共主循环：engine/loop.py

```python
# engine/loop.py

def batch_generate_loop(model, sampler, scheduler, adapter, prompts,
                        max_new_tokens, eos_token_id=None, device="cpu",
                        metrics=None):
    """公共 batch generation 主循环。对齐 vLLM V1 execute_model 模式。"""

    # ── 初始化：绑定 cache 到模型的每个 Attention 层 ──
    adapter.bind_kv_cache(model.model)  # model.model = Qwen3Model

    results = {}

    while scheduler.has_unfinished():
        # ── 1. Admit + Allocate ──
        admitted = []
        while scheduler.waiting:
            req = scheduler.waiting[0]
            if not adapter.can_admit(req):
                break
            scheduler.waiting.popleft()
            req.status = RequestStatus.RUNNING
            scheduler.running[req.request_id] = req
            adapter.allocate(req)
            admitted.append(req)

        # ── 2. Batched Prefill ──
        if admitted:
            request_ids = [r.request_id for r in admitted]
            prompt_lens = [r.prompt_ids.shape[1] for r in admitted]
            max_len = max(prompt_lens)

            # 构造 padded batch
            input_ids = torch.zeros(len(admitted), max_len, dtype=torch.long, device=device)
            positions = torch.zeros(len(admitted), max_len, dtype=torch.long, device=device)
            for i, req in enumerate(admitted):
                plen = req.prompt_ids.shape[1]
                input_ids[i, :plen] = req.prompt_ids.squeeze(0)
                positions[i, :plen] = torch.arange(plen, device=device)

            # 构造 metadata 并执行
            metadata = adapter.make_prefill_metadata(admitted)
            with set_forward_context(metadata):
                hidden_states = model(input_ids, positions=positions)
            logits = model.compute_logits(hidden_states)

            # 逐请求采样第一个 decode token
            for i, req in enumerate(admitted):
                plen = prompt_lens[i]
                token = sampler(logits[i, plen - 1, :].unsqueeze(0))
                req.last_token = token.unsqueeze(0)
                req.generated_tokens.append(req.last_token)
                req.num_generated = 1
                req.seq_len = plen

        # ── 3. Decode ──
        running = list(scheduler.running.values())
        if not running:
            break

        # 3a. prepare_decode：分配 cache 空间（不 +1）
        for req in running:
            adapter.prepare_decode(req.request_id)

        # 3b. 构造 decode batch
        request_ids = [r.request_id for r in running]
        next_tokens = torch.cat([r.last_token for r in running], dim=0)
        positions = torch.tensor(
            adapter.decode_positions(request_ids),
            dtype=torch.long, device=device,
        ).unsqueeze(1)

        # 3c. forward
        metadata = adapter.make_decode_metadata(running)
        with set_forward_context(metadata):
            hidden_states = model(next_tokens, positions=positions)
        logits = model.compute_logits(hidden_states)

        # 3d. commit_decode：seq_len +1
        adapter.commit_decode(request_ids)

        # 3e. sample + finish
        sampled = sampler(logits[:, -1, :])
        for req, token in zip(running, sampled):
            req.last_token = token.unsqueeze(0)
            req.generated_tokens.append(req.last_token)
            req.num_generated += 1
            req.seq_len += 1

            is_max = req.num_generated >= req.max_new_tokens
            is_eos = eos_token_id is not None and token.item() == eos_token_id
            if is_max or is_eos:
                scheduler.mark_finished(req)
                adapter.free(req.request_id)
                results[req.request_id] = req

    # ── 收集结果 ──
    output = []
    for rid in sorted(results.keys(), key=int):
        req = results[rid]
        output.append(torch.cat([req.prompt_ids] + req.generated_tokens, dim=1))
    return output
```

### C. CacheAdapter 实现：cache/adapter.py

```python
# cache/adapter.py

class BatchedCacheAdapter:
    """M3 fixed-slot BatchedKVCache 的 adapter。"""

    def __init__(self, cache: BatchedKVCache, max_num_seqs: int):
        self.cache = cache
        self.max_num_seqs = max_num_seqs

    def can_admit(self, request) -> bool:
        return len(self.cache.occupied_slots) < self.max_num_seqs

    def allocate(self, request) -> None:
        slot = self.cache.allocate_slot(request.request_id)
        request.slot_id = slot

    def free(self, request_id: str) -> None:
        self.cache.free_slot(request_id)

    # ── decode 三步时序 ──

    def prepare_decode(self, request_id: str) -> None:
        pass  # fixed-slot 已在 prefill 时分配

    def decode_positions(self, request_ids: list[str]) -> list[int]:
        return [int(self.cache.seq_lens[self.cache.slot_map[rid]])
                for rid in request_ids]

    def commit_decode(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            slot = self.cache.slot_map[rid]
            self.cache.seq_lens[slot] += 1

    # ── 元数据构造 ──

    def make_prefill_metadata(self, requests) -> AttentionMetadata:
        slots = [self.cache.slot_map[r.request_id] for r in requests]
        prompt_lens = [r.prompt_ids.shape[1] for r in requests]
        # slot_mapping: 每个 token 对应的 (slot, position) 的展平索引
        # 用于 attention 层写入 cache
        total_tokens = sum(prompt_lens)
        slot_mapping = self._build_slot_mapping(slots, prompt_lens)
        return AttentionMetadata(
            num_seqs=len(requests),
            seq_lens=torch.tensor(prompt_lens),
            slot_mapping=slot_mapping,
        )

    def make_decode_metadata(self, requests) -> AttentionMetadata:
        slots = [self.cache.slot_map[r.request_id] for r in requests]
        seq_lens = [int(self.cache.seq_lens[s]) for s in slots]
        # decode 时每个 token 的 slot_mapping = (slot, seq_len)
        slot_mapping = torch.tensor(
            [s * self.cache.max_seq_len + l for s, l in zip(slots, seq_lens)])
        return AttentionMetadata(
            num_seqs=len(requests),
            seq_lens=torch.tensor(seq_lens),
            slot_mapping=slot_mapping,
        )

    # ── cache 绑定 ──

    def bind_kv_cache(self, qwen3_model) -> None:
        """将每层的 LayerKVCache 绑定到对应的 Attention 层。"""
        for i, layer in enumerate(qwen3_model.layers):
            layer.self_attn.attn.bind_kv_cache(self.cache.layers[i])

    def _build_slot_mapping(self, slots, prompt_lens):
        mapping = []
        for slot, plen in zip(slots, prompt_lens):
            for pos in range(plen):
                mapping.append(slot * self.cache.max_seq_len + pos)
        return torch.tensor(mapping)


class PagedCacheAdapter:
    """M4 paged PagedKVCache 的 adapter。"""

    def __init__(self, cache: PagedKVCache):
        self.cache = cache

    def can_admit(self, request) -> bool:
        prompt_len = request.prompt_ids.shape[1]
        return self.cache.can_allocate(prompt_len)

    def allocate(self, request) -> None:
        prompt_len = request.prompt_ids.shape[1]
        self.cache.allocate_request(request.request_id, prompt_len)

    def free(self, request_id: str) -> None:
        self.cache.free_request(request_id)

    # ── decode 三步时序 ──

    def prepare_decode(self, request_id: str) -> None:
        """分配新 block（如果需要），但不 +1。"""
        table = self.cache.block_tables[request_id]
        if table.needs_new_block():
            block_id = self.cache.block_pool.allocate()
            table.append_block(block_id)

    def decode_positions(self, request_ids: list[str]) -> list[int]:
        """seq_len 未 +1，直接就是正确的 0-indexed position。"""
        return [self.cache.block_tables[rid].seq_len for rid in request_ids]

    def commit_decode(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            self.cache.block_tables[rid].extend(1)

    # ── 元数据构造 ──

    def make_prefill_metadata(self, requests) -> AttentionMetadata:
        request_ids = [r.request_id for r in requests]
        prompt_lens = [r.prompt_ids.shape[1] for r in requests]
        block_table = self._build_block_table(request_ids)
        return AttentionMetadata(
            num_seqs=len(requests),
            seq_lens=torch.tensor(prompt_lens),
            block_table=block_table,
        )

    def make_decode_metadata(self, requests) -> AttentionMetadata:
        request_ids = [r.request_id for r in requests]
        seq_lens = [self.cache.block_tables[rid].seq_len for rid in request_ids]
        block_table = self._build_block_table(request_ids)
        return AttentionMetadata(
            num_seqs=len(requests),
            seq_lens=torch.tensor(seq_lens),
            block_table=block_table,
        )

    # ── cache 绑定 ──

    def bind_kv_cache(self, qwen3_model) -> None:
        """将整个 PagedKVCache 绑定到每个 Attention 层。
        Attention 层通过 layer_idx 访问自己那层的 KV 数据。
        """
        for i, layer in enumerate(qwen3_model.layers):
            layer.self_attn.attn.bind_kv_cache(self.cache, layer_idx=i)

    def _build_block_table(self, request_ids):
        max_blocks = max(len(self.cache.block_tables[rid].block_ids)
                         for rid in request_ids)
        table = torch.zeros(len(request_ids), max_blocks, dtype=torch.long)
        for i, rid in enumerate(request_ids):
            ids = self.cache.block_tables[rid].block_ids
            table[i, :len(ids)] = torch.tensor(ids)
        return table
```

### D. 模型链：model/qwen3.py + model/attention.py

```python
# model/attention.py

class Attention(nn.Module):
    """独立的 attention 层。对齐 vLLM V1 的 Attention 类。

    KV cache 通过 bind_kv_cache 绑定。
    Attention metadata 通过 get_forward_context().attn_metadata 获取。
    forward 只接收 (query, key, value)。
    """
    def __init__(self, num_heads, num_kv_heads, head_dim, scaling):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.num_key_value_groups = num_heads // num_kv_heads
        self.kv_cache = None      # bind_kv_cache 后赋值
        self.layer_idx = None     # bind_kv_cache 后赋值（paged 路径需要）

    def bind_kv_cache(self, kv_cache, layer_idx=None):
        self.kv_cache = kv_cache
        self.layer_idx = layer_idx

    def forward(self, query, key, value):
        """query/key/value: [num_tokens, num_heads * head_dim] 或已 reshape 的形式。

        1. 将 q/k/v reshape 为 [B, T, heads, head_dim] → transpose 为 [B, heads, T, D]
        2. 写 cache（如果有 kv_cache）
        3. 从 cache 读完整历史 K/V
        4. GQA repeat_kv
        5. attention core (scores + mask + softmax + matmul)
        """
        metadata = get_forward_context().attn_metadata

        # reshape
        B = metadata.num_seqs
        q = query.transpose(1, 2)   # [B, n_q, T, D]
        k = key.transpose(1, 2)     # [B, n_kv, T, D]
        v = value.transpose(1, 2)   # [B, n_kv, T, D]

        # 写 cache + 读完整历史 K/V
        if self.kv_cache is not None:
            k, v, valid_lens = self._cache_rw(k, v, metadata)
        else:
            valid_lens = None

        # GQA repeat_kv
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # attention core
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        # causal mask
        T = q.shape[2]
        if T > 1:
            seq_k = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(T, seq_k, dtype=torch.bool, device=q.device),
                diagonal=metadata.seq_lens[0] - T + 1 if self.kv_cache else 1,
            )
            attn_weights.masked_fill_(causal_mask[None, None, :, :],
                                       torch.finfo(attn_weights.dtype).min)
        # valid_lens mask
        if valid_lens is not None:
            attn_weights = self._apply_valid_lens_mask(attn_weights, valid_lens)

        attn_weights = torch.softmax(attn_weights, dim=-1,
                                      dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        # [B, heads, T, D] → [B, T, heads * D]
        return attn_output.transpose(1, 2).contiguous().view(B, T, -1)

    def _cache_rw(self, k, v, metadata):
        """根据 kv_cache 类型分发到具体读写逻辑。"""
        if isinstance(self.kv_cache, LayerKVCache):
            # M3 路径：BatchedLayerKVCache
            return self._batched_cache_rw(k, v, metadata)
        elif isinstance(self.kv_cache, PagedKVCache):
            # M4 路径：PagedKVCache
            return self._paged_cache_rw(k, v, metadata)
        else:
            # M2 路径：单序列 LayerKVCache
            return self._single_cache_rw(k, v, metadata)


# model/qwen3.py

class Qwen3Attention(nn.Module):
    """QKV projection + QK norm + RoPE + 调用独立 Attention 层。"""

    def __init__(self, config):
        super().__init__()
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings)
        self.attn = Attention(
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            scaling=1.0 / (config.head_dim ** 0.5),
        )

    def forward(self, positions, hidden_states):
        """对齐 vLLM V1 Qwen3Attention.forward(positions, hidden_states)。"""
        # QKV projection
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # reshape for multi-head
        B, T, _ = hidden_states.shape
        q = q.view(B, T, self.attn.num_heads, self.attn.head_dim)
        k = k.view(B, T, self.attn.num_kv_heads, self.attn.head_dim)
        v = v.view(B, T, self.attn.num_kv_heads, self.attn.head_dim)
        # QK norm
        q = self.q_norm(q)
        k = self.k_norm(k)
        # RoPE（per-layer，对齐 vLLM）
        cos, sin = self.rotary_emb(q, positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        # Attention（cache 操作在 Attention.forward 内部）
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(self, positions, hidden_states):
        """对齐 vLLM V1 Qwen3DecoderLayer.forward(positions, hidden_states)。"""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)

    def forward(self, input_ids, positions):
        """对齐 vLLM V1 Qwen3Model.forward(input_ids, positions)。"""
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions):
        """对齐 vLLM V1 Qwen3ForCausalLM.forward(input_ids, positions)。"""
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states):
        """logits 计算与 model forward 分离（对齐 vLLM V1）。"""
        return self.lm_head(hidden_states)
```

### E. Protocol：engine/protocol.py

```python
class LLMModel(Protocol):
    """最小 LLM 推理协议（对齐 vLLM V1）。"""
    def __call__(self, input_ids, *, positions=None) -> torch.Tensor: ...
    def compute_logits(self, hidden_states) -> torch.Tensor: ...
```

### F. 调用链全景图

```
paged_core.py (30 行)
    │  创建 PagedCacheAdapter + 调 batch_generate_loop
    ▼
batch_generate_loop (loop.py, ~120 行)
    │  adapter.bind_kv_cache(model.model)
    │  set_forward_context(metadata)
    │  model(input_ids, positions=positions)
    ▼
Qwen3ForCausalLM.forward(input_ids, positions=*)         # 2 参数
    │  self.model(input_ids, positions)
    ▼
Qwen3Model.forward(input_ids, positions)                # 2 参数
    │  for layer in self.layers: layer(positions, h)
    ▼
Qwen3DecoderLayer.forward(positions, hidden_states)     # 2 参数
    │  self.self_attn(positions, hidden_states)
    ▼
Qwen3Attention.forward(positions, hidden_states)        # 2 参数
    │  QKV proj + QK norm + RoPE
    │  self.attn(q, k, v)
    ▼
Attention.forward(query, key, value)                    # 3 参数
    │  metadata = get_forward_context().attn_metadata   # 从 context 取
    │  self.kv_cache (bind_kv_cache 绑定的)              # 从 self 取
    │  cache_rw → attention core → output
    ▼
返回 [B, T, hidden_size]
```
