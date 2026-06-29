# M2 KV Cache Benchmark — Qwen3-0.6B

**目的**：对比 M1（无 KV Cache）和 M2（有 KV Cache）在不同 prompt 长度下的生成速度，展示 KV Cache 加速效果随序列长度增长的变化趋势。

## 运行环境

| 项目 | 值 |
|------|-----|
| 硬件 | MacBook Pro M3 Pro |
| Device | mps |
| dtype | bfloat16 |
| 模型 | Qwen3-0.6B |
| gen_tokens | 128 |
| warmup | 1 |
| 日期 | 2026-06-29 |

## 结果

```
prompt_tokens   M1 tok/s   M2 tok/s   Speedup
-----------------------------------------------
           32       13.8       24.9     1.80x
           64       12.7       28.9     2.27x
          128        9.6       25.7     2.67x
          256        6.2       24.1     3.91x
          512        3.3       24.1     7.36x
```

## 分析

- **M1 tok/s**：随 prompt 增长持续下降。每步 decode 要对所有历史 token 重算 Attention，复杂度 O(T²)。prompt=512 时已跌到 3.3 tok/s，几乎不可用。
- **M2 tok/s**：基本稳定在 24-29 tok/s。decode 阶段每步只传 1 个 token，Attention 只算当前 token 对所有历史 token 的点积，复杂度 O(T)，但 T 在 decode 阶段是常数（1）。
- **Speedup**：从 1.80x（prompt=32）到 7.36x（prompt=512），随 prompt 线性增长，符合理论预期。

## 复现命令

```bash
uv run python scripts/bench_kv_cache.py \
  --model-dir ~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B \
  --device mps --dtype bf16 \
  --gen-tokens 128
```
