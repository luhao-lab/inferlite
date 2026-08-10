# M5-T4 — E2E 测试 + 回归 + Benchmark

> 验证 prefix cache 端到端正确性、M4 回归、以及跳过 prefill 的性能收益。

## 元信息

| 字段 | 内容 |
|---|---|
| 任务 ID | M5-T4 |
| 状态 | ⬜ pending |
| 前置 | M5-T3 ✅ |
| 后续 | M5-T5 Docs & Tag |
| 估时 | 2～3h |
| 核心文件 | `tests/` |

## 范围

### 1. Prefix Cache 命中测试

```python
# 相同 prompt 的两个请求
prompts = [token_ids_a, token_ids_a]  # 完全相同
results = batch_generate_paged(model, sampler, prompts, ...)
# 验证：
# - 第二个请求的 num_cached_blocks > 0
# - 两个请求输出相同（torch.equal）
```

### 2. Prefix Cache 部分命中测试

```python
# 相同前缀 + 不同后缀
prompts = [token_ids_a, token_ids_a_prefix + different_suffix]
# 验证：
# - 第二个请求命中前缀部分的 block
# - 输出正确（与 serial 等价）
```

### 3. LRU 淘汰测试

```python
# 填满 block pool 后，验证 LRU 淘汰顺序
# - 最久未用的 cached block 先被淘汰
# - touch 后 LRU 顺序变化
```

### 4. CoW 测试

```python
# shared block 写入后不污染其他请求
# - 请求 A 和 B 共享前缀 block
# - A 继续生成，触发 CoW
# - B 的 cached block 不受影响
```

### 5. M4 回归

```python
# 所有 M4 测试在 prefix cache 关闭时仍通过
# - test_paged_attention 系列
# - test_paged_batch_engine 系列
# - test_batch_generate 系列
```

### 6. Benchmark（可选）

| 指标 | 方法 |
|---|---|
| prefix cache hit rate | 相同 prompt 重复请求，统计命中 block 数 / 总 block 数 |
| prefill 跳过率 | cached tokens / total prompt tokens |
| TTFT 对比 | 有 prefix cache vs 无 prefix cache 的 first token latency |

## DoD

- [ ] 相同 prompt 第二个请求命中 prefix cache
- [ ] 跳过 prefill 后输出 `torch.equal` 等价
- [ ] LRU 淘汰顺序正确
- [ ] CoW 不污染 shared block
- [ ] M4 全量回归通过
- [ ] （可选）benchmark 数据归档
