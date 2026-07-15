# M3-T5 E2E Correctness

> M3 第五张任务卡：用端到端测试证明 continuous batching 没有改变语义。

## 元信息
- **任务 ID**: M3-T5
- **里程碑**: M3 — Continuous Batching
- **状态**: ✅ done
- **前置**: M3-T4
- **估时**: 3h

## 目标

**要解决什么问题**：

Continuous batching 改变的是服务执行方式，不应该改变每个请求自己的生成语义。

本卡要回答：

```text
同一组请求：
逐条串行 generate 的结果
是否等价于
BatchEngine continuous batching 的结果？
```

**做完是什么效果**：

在 deterministic sampler / fake model 下：

```python
serial_outputs = [generate(engine, p, max_new_tokens=...) for p in prompts]
batch_outputs = batch_generate(engine, prompts, max_new_tokens=..., max_num_slots=...)
assert batch_outputs == serial_outputs
```

并验证短请求不会被长请求阻塞到固定 batch 结束。

**不做什么（边界）**：

- 不以性能为主，性能放到 T6。
- 不做真实服务压测。
- 不做随机采样一致性测试。
- 不证明浮点 bitwise 完全一致；真实模型可用 close 或 deterministic fake model。

## 产出文件

- `tests/e2e/test_batch_generate.py`
- `tests/e2e/test_continuous_batching_trace.py`
- 必要时新增测试用 fake model / fake sampler fixture

## 算法核心

### 1. 串行 vs batch 语义等价

```python
@pytest.mark.parametrize("max_num_slots", [1, 2, 4])
def test_batch_generate_matches_serial(engine, prompts, max_num_slots):
    serial = [generate(engine, p[None, :], max_new_tokens=8) for p in prompts]
    batched = batch_generate(engine, prompts, max_new_tokens=8, max_num_slots=max_num_slots)
    assert_tokens_equal(batched, serial)
```

### 2. continuous batching trace

用 fake model 控制不同请求结束时刻：

```text
req_a: 2 tokens finished
req_b: 5 tokens finished
req_c: 3 tokens finished
max_num_slots = 2
```

期望 trace：

```text
step 0: running [a, b]
step 1: running [a, b]
step 2: a finished, slot released, c admitted at next boundary
step 3: running [b, c]
...
```

核心不是具体 token，而是证明：

```text
短请求完成后立即释放 slot，等待请求在下一轮进入。
```

这个 trace 是 M3 区分 static batching 的关键：不能只证明 `batch_size > 1`，还要证明 finished 请求不会锁住 wave，slot 可以跨请求复用。

### 3. slot 复用不串数据

构造 req_a 完成后 slot 被 req_c 复用，验证 req_c 不读到 req_a 的旧 KV：

```python
assert req_c_output == serial_req_c_output
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|---|---|---|
| 1 | `max_num_slots=1` | 等价串行执行 | 精确 |
| 2 | `max_num_slots>1` | 等价串行逐条 generate | 精确或 close |
| 3 | prompt 长度不同 | 输出语义不变 | 精确或 close |
| 4 | output 长度不同 | 短请求提前完成 | trace 精确 |
| 5 | slot 复用 | 新请求不受旧 KV 污染 | 精确 |
| 6 | EOS 早停 | 输出停在 EOS 后 | 精确 |
| 7 | waiting queue 大于 slots | 最终全部完成 | 精确 |
| 8 | batch size trace | 每轮 running 数符合预期 | 精确 |
| 9 | 非 static wave | finished 请求不锁住 batch wave | trace 精确 |
| 10 | waiting 不占资源 | pending 请求未提前 allocate slot | 精确 |

## DoD

- [ ] E2E 测试覆盖串行 vs continuous batching 等价。
- [ ] E2E 测试覆盖可变 prompt 长度。
- [ ] E2E 测试覆盖可变 output 长度。
- [ ] trace 测试证明不是 static batching。
- [ ] trace 测试证明 waiting 请求不提前占 KV slot。
- [ ] trace 测试证明 finished 后 slot 可在下一轮跨请求复用。
- [ ] slot 复用无 KV 污染测试通过。
- [ ] `uv run pytest tests/e2e/test_batch_generate.py tests/e2e/test_continuous_batching_trace.py -q` 通过。
- [ ] 全量 `uv run pytest` 通过。
- [ ] commit `test(engine): add continuous batching e2e correctness tests (M3-T5 done)`。

## 坑（按概率排序）

1. **用随机采样导致结果不可复现**：E2E correctness 应使用 greedy 或 deterministic sampler。
2. **真实模型浮点差异误判**：可先用 fake model 做精确测试，再用真实模型做 smoke test。
3. **只测吞吐不测语义**：M3 先保证每个请求独立语义不变。
4. **trace 不够可观测**：BatchEngine 需要提供 batch trace 或 debug hook。

## 完成总结

### 测试覆盖

12 个 E2E 测试，190/190 全量回归通过：

**串行 vs batch 语义等价**（`test_batch_generate.py`）：
- DeterministicModel：max_num_slots=1/2/4 三档，验证 token 级 `torch.equal`
- 真实 Qwen3ForCausalLM：3 个不同长度 prompt，验证真实模型 token 级一致
- 变长 prompt、EOS 早停、waiting>slots 全部完成

**continuous batching trace**（`test_continuous_batching_trace.py`）：
- 不同 output 长度、slot 复用无 KV 污染、batch size trace
- 非 static batching（finished 请求不锁住 wave）、waiting 不占 slot
- EOS trace 验证 batch size 变化

### 关键发现

真实模型测试证明：M3 的所有改动（BatchedKVCache + prefill/decode 分派 + per-row mask + gather）**只有性能变化，语义完全不变**——serial generate 和 batch_generate 在 token 级别 `torch.equal`。

### 修改文件

| 文件 | 改动 |
|---|---|
| `tests/e2e/test_batch_generate.py` | 串行 vs batch 等价测试（含真实模型） |
| `tests/e2e/test_continuous_batching_trace.py` | continuous batching trace 测试 |
| `inferlite/model/attention.py` | prefill 分派 + `_batched_prefill_rw` 方法 |
| `inferlite/model/qwen3.py` | `Qwen3Model.forward` 加 `position_ids.dim()==1` 分支 |
| `inferlite/engine/batch_core.py` | `cache_positions` 用 `[B]`，`position_ids` 单独 unsqueeze |
