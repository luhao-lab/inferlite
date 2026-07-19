# M3 Continuous Batching Benchmark — Qwen3-0.6B

**目的**：对比 M2 serial baseline 和 M3 continuous batching，拆解 prefill/decode/TTFT/ITL 指标，定位 M3 的性能特征。

## 运行环境

| 项目 | 值 |
|------|-----|
| 硬件 | MacBook Pro M3 Pro |
| Device | mps |
| dtype | bfloat16 |
| 模型 | Qwen3-0.6B |
| num_requests | 4 |
| max_new_tokens | 16 |
| prompt_len | 32 |
| warmup | 1 |
| 日期 | 2026-07-18 |

## 结果

### 主对比（max_num_slots=2 vs serial）

```
────────────────────────────────────────
A. Serial baseline (max_num_slots=1)
────────────────────────────────────────
  total_ms:            1827.42
  output_tokens_per_s: 35.02
  tpot_ms:             28.55
  total_output_tokens: 64
────────────────────────────────────────
B. Continuous batching (max_num_slots=2)
────────────────────────────────────────
  prefill_ms_p50:      24.60
  decode_step_ms_p50:  130.80
  ttft_ms_p50:         1582.29
  itl_ms_p50:          145.60
  output_tokens_per_s: 15.29
  tpot_ms:             65.40
  avg_batch_size:      2.00
  slot_utilization:    0.93
  total_decode_ms:     3924.03
  total_output_tokens: 60
────────────────────────────────────────
Comparison
────────────────────────────────────────
  serial throughput:  35.02 tok/s
  batch throughput:   15.29 tok/s
  speedup:            0.44x
```

### 消融实验（max_num_slots=1 vs serial）

为定位瓶颈来源，单独测 B=1 的 batch_generate（消除 batch 维影响）：

```
────────────────────────────────────────
A. Serial baseline (max_num_slots=1)
────────────────────────────────────────
  total_ms:            1807.92
  output_tokens_per_s: 35.40
  tpot_ms:             28.25
────────────────────────────────────────
B. Continuous batching (max_num_slots=1)
────────────────────────────────────────
  prefill_ms_p50:      51.75
  decode_step_ms_p50:  75.12
  ttft_ms_p50:         1913.78
  itl_ms_p50:          79.22
  output_tokens_per_s: 13.31
  tpot_ms:             75.12
  avg_batch_size:      1.00
  slot_utilization:    0.93
────────────────────────────────────────
Comparison
────────────────────────────────────────
  serial throughput:  35.40 tok/s
  batch throughput:   13.31 tok/s
  speedup:            0.38x
```

## 分析

### 结论：M3 batch_generate 在纯 PyTorch + MPS 上比 M2 serial 慢

| 场景 | serial tpot | batch tpot | speedup |
|------|-------------|------------|---------|
| B=1 | 28.25 ms | 75.12 ms | 0.38x |
| B=2 | 28.55 ms | 65.40 ms | 0.44x |

**关键观察**：B=1 时 batch_generate 已经比 serial 慢 2.7 倍（tpot 75ms vs 28ms），说明瓶颈不在 batch 维，而在 **M3 cache 路径本身**。

### 瓶颈定位：分段计时 micro-benchmark

为确定主因，在 `_batched_cache_rw` 内加分段计时（B=1, 28 层, 60 步 decode）：

| 段 | 代码 | 总耗时 | 占比 | 每层每步 |
|------|------|--------|------|----------|
| tolist | `cache_slots.tolist()` | 3.7 ms | 0.1% | 0.002 ms |
| **write** | `for i: cache.k[slot, :, pos, :] = k[i]` | **2447.9 ms** | **63.0%** | **1.457 ms** |
| max_item | `cache_positions.max().item()` | 566.5 ms | 14.6% | 0.337 ms |
| gather | `cache.k[cache_slots, :, :max_len]` | 868.2 ms | 22.3% | 0.517 ms |
| **Total** | `_batched_cache_rw` | 3886.4 ms | 100% | 64.77 ms/step |

**主因不是 gather（22%），是 for 循环写 cache（63%）**。这个发现颠覆了最初"fancy index gather 是主因"的推测。

**关键认知**：三个瓶颈（write 63% + gather 22% + max_item 15% = 100%）**全是 cache 读写路径，不在 attention 计算**。M2 和 M3 的 attention 计算（`q @ k^T` + softmax + `attn @ v`）完全一样，性能差异 100% 来自 cache 读写路径。

### 三个性能开销来源（按实测占比排序）

1. **for 循环写 cache**（63%，主因）
   ```python
   for i, slot in enumerate(cache_slots.tolist()):
       pos = int(cache_positions[i])
       cache.k[slot, :, pos : pos + 1, :] = k[i]
       cache.v[slot, :, pos : pos + 1, :] = v[i]
   ```
   28 层 × 60 步 = 1680 次循环调用，每次写 `[H_kv, 1, D]` 的 slice。B=1 也要跑（1 次迭代，但循环开销 + slice 赋值 + MPS kernel launch）。

2. **fancy indexing gather**（22.3%）
   ```python
   k = cache.k[cache_slots, :, :max_len, :]
   ```
   advanced indexing 触发 copy，每次 decode 都重新 gather 整个历史。

3. **`.item()` 同步**（14.6%）
   ```python
   max_len = int(cache_positions.max().item()) + 1
   ```
   `.item()` 触发 GPU→CPU 同步，每层每步都调一次（28 × 60 = 1680 次）。

### 为什么 M2 的 `KVCache` 快

```python
# M2 _single_cache_rw（无循环、无 gather、无 .item()）
cache.k[:, :, cache_position : cache_position + seq_len, :] = k   # 一次性写
k = cache.k[:, :, : cache_position + seq_len, :]                 # 切片 view，零拷贝
```

M2 的 cache 第 0 维是 batch（固定 1），切片是 view，零拷贝。M3 的 `BatchedKVCache` 第 0 维是 slot，必须用 fancy index 选 slot + for 循环写，触发 copy + Python 开销。

### M2 vs M3：结构同构，访问方式不同

| | M2 LayerKVCache | M3 BatchedLayerKVCache |
|---|---|---|
| cache tensor shape | `[1, H_kv, L, D]` | `[S, H_kv, L, D]` |
| 第 0 维语义 | batch（固定 1） | slot（多请求复用） |
| 写第 0 维 | `[:, :, ...]` 切片 | `[slot, :, ...]` 指定 slot |
| 写 T 维 | `[:, :, pos:pos+T, :]` 一次性 | `for i: [slot, :, pos, :] = k[i]` 循环 |
| 读 T 维 | `[:, :, :len]` view（零拷贝） | `[cache_slots, :, :max_len]` fancy index copy |

**核心矛盾**：continuous batching 要求 per-row 独立位置（不同 slot、不同 pos），M2 的切片路径无法表达这种语义，必须用循环 + gather。这是 batched 的固有代价。

### 为什么 B=2 的 speedup（0.44x）略高于 B=1（0.38x）

B=2 的单步 decode（130ms）≈ 2× B=1 的单步 decode（75ms），说明 MPS 上 B=2 的矩阵乘有部分摊销。但 gather + for 循环开销也在（B=2 也要 gather + 循环 2 次），且短请求 padding 浪费。综合下来还是比 serial 慢。

### 理论上 batch 多大能追平 serial

假设 M3 的单步 decode 耗时 = `a + b × B`（a 是固定开销，b 是每行增量）：

```text
B=1: 75 = a + b
B=2: 130 = a + 2b
解得: a = 20, b = 55
```

tpot = (a + b×B) / B = a/B + b

要 tpot < 28ms（serial），需要 `a/B + b < 28`，即 `20/B + 55 < 28`，**不可能**（b=55 已经大于 28）。纯 PyTorch 的 M3 在 MPS 上永远比 M2 serial 慢，不管 B 多大。

## 与主流框架的对比

| 框架 | batched decode 实现 | for 循环写 | gather | .item() 同步 | 是否有此问题 |
|------|---------------------|-----------|--------|--------------|--------------|
| M2 inferlite | `KVCache` 切片 view | 否 | 否 | 否 | 否（但只支持单请求） |
| M3 inferlite | `BatchedKVCache` PyTorch gather | ✅ 63% | ✅ 22% | ✅ 15% | 是（纯 PyTorch，不调 kernel） |
| nano-vllm | **Triton store kernel + Flash Attention** | **否（Triton kernel）** | **否（Flash Attention paged）** | 否 | **否**（≈ vLLM 性能） |
| vLLM | PagedAttention CUDA kernel | 否（kernel 内） | 否（kernel 内） | 否 | 否 |
| SGLang | RadixAttention + 自定义 kernel | 否 | 否 | 否 | 否 |

**关键认知修正**：nano-vllm **没有此限制**。它的 1200 行 Python 代码调用的是 **Triton kernel**（`store_kvcache_kernel`）和 **Flash Attention 库**（`flash_attn_with_kvcache`），不是纯 PyTorch。官方 benchmark 显示 nano-vllm ≈ vLLM（1434 vs 1362 tok/s, RTX 4070 Laptop）。

**inferlite M3 的性能限制是"纯 PyTorch + 不调 kernel"的路线选择，不是教学版固有代价**。nano-vllm 选择调 Flash Attention + 写 Triton，所以性能能持平 vLLM。

## M3 的教学目标定位

M3 的核心收益是 **continuous batching 语义**（请求进退 + slot 复用），不是性能：

- ✅ 短请求完成后释放 slot，等待请求下一轮进入（L0-9 非 static wave 测试通过）
- ✅ 不同长度的请求可以并发 decode（L0-4 测试通过）
- ✅ slot 复用无 KV 污染（L0-5 测试通过）
- ✅ serial vs batch 语义等价（L0-1/L0-2 测试通过，token 级 torch.equal）

性能优化留给后续里程碑（按瓶颈对应 + 硬件可用性排序）：

- **M4 PagedAttention（PyTorch 伪版）**：block_table + 按需 gather，避免 materialize 整个 dense tensor。
  - 对应瓶颈：gather（22%）、max_item（15%，元数据一次算传所有层）
  - Mac 可用：✅
  - 局限：for 循环写 cache（63%）还在，仍 Python
- **M4 + torch.compile**（Mac 友好的附加优化）：编译融合算子，部分消除 Python 循环开销。
  - Mac 可用：✅
  - 教学价值：中（black box 编译器）
- **M8 Triton kernel**：用 Triton 写 `store_kvcache_kernel`（替代 for 循环写）+ 调 Flash Attention（替代 PyTorch gather + per-row mask）。
  - 对应瓶颈：全部三个（write 63% + gather 22% + max_item 15%）
  - **Mac 不可用**：Triton 和 Flash Attention 都只支持 NVIDIA CUDA，MPS 跑不了
  - 彻底解决后性能可接近 vLLM（参考 nano-vllm: 1434 vs vLLM 1362 tok/s）

**Mac 用户的加速路径**：M3（纯 PyTorch，慢）→ M4（paged gather，部分缓解）→ M4 + torch.compile（编译融合，再快一点）→ M8 时需 NVIDIA GPU 或用 flex_attention / Metal kernel 替代（生态不成熟，教学价值有限）。

## 局限性

- 教学版不做 MPS/CUDA 同步，时间偏小
- 合成 pad prompt 非真实分布
- 每组只跑一次，有随机波动
- max_num_slots=1 的 batch_generate 路径含 queue 等待（TTFT 1913ms 包含 3 个请求在 waiting 排队）

## 复现命令

```bash
# 主对比
uv run python scripts/bench_continuous_batching.py \
  --model-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \
  --device mps --dtype bf16 \
  --num-requests 4 --max-num-slots 2 \
  --max-new-tokens 16 --prompt-len 32

# 消融（B=1）
uv run python scripts/bench_continuous_batching.py \
  --model-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \
  --device mps --dtype bf16 \
  --num-requests 4 --max-num-slots 1 \
  --max-new-tokens 16 --prompt-len 32
```
