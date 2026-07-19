# M3-T4 BatchEngine

> M3 第四张任务卡：串起 scheduler、slot cache、batched attention，形成 continuous batching 的最小 engine。

## 元信息
- **任务 ID**: M3-T4
- **里程碑**: M3 — Continuous Batching
- **状态**: ✅ done
- **前置**: M3-T1, M3-T2, M3-T3
- **估时**: 5h

## 目标

**要解决什么问题**：

T1/T2/T3 分别解决了状态机、KV slot、batched attention。T4 要把它们串成真正的 M3 执行流：

```text
submit requests
  ↓
prefill one by one
  ↓
admit to running slots
  ↓
each decode iteration forms current batch
  ↓
batched decode one token
  ↓
finished requests leave, waiting requests enter
```

M3 的核心策略是：

```text
prefill: one request at a time
decode: batch multiple running requests every iteration
```

**做完是什么效果**：

可以通过一个同步 API 执行多请求生成：

```python
outputs = batch_generate(
    engine,
    prompts=[ids_a, ids_b, ids_c],
    max_new_tokens=16,
    max_num_slots=2,
)
assert len(outputs) == 3
```

其中 slot 数为 2 时，第 3 个请求会先 waiting，等前面请求 finished 后再进入 running。

**不做什么（边界）**：

- 不做异步 streaming server。
- 不做 prefill batching。
- 不做 chunked prefill。
- 不做抢占/换出。
- 不做 token budget scheduler。
- 不做 decode-first + chunked prefill 生产策略。
- 不做 prefill/decode mixed batch。
- 不做复杂采样策略重构。

## 产出文件

- `inferlite/engine/batch_core.py::batch_generate`
- `inferlite/model/qwen3.py` — `Qwen3ForCausalLM.forward()` 加 `cache_slots`/`cache_positions` 透传
- `tests/unit/test_batch_engine.py`

## 设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| BatchEngine vs 复用 EngineCore | 纯函数 `batch_generate` | EngineCore.step() 不传 kv_cache/position_ids/cache_slots，复用无意义；纯函数和 M2 generate() 对称 |
| seq_len 语义 | 下一个写入位置 | 和 M2 cur_len 一致；nano-vllm 也这么做 |
| prefill 后 seq_len | = prompt_len | prefill 写 KV[0..prompt_len-1]，下一步写 prompt_len |
| batch_generate 位置 | `engine/batch_core.py` | 和 `engine/core.py` 对称 |
| batch_generate 参数 | 直接调 `model()` + `sampler()` | EngineCore.step() 不支持 M3 参数，主流框架也是 engine 直接持有 model |
| 主循环结构 | finish → admit → decode | iteration-level scheduling，不做 prefill batching |

## 实现步骤

### Step 1: qwen3.py — Qwen3ForCausalLM.forward 加参数

1. `forward()` 签名加 `cache_slots`, `cache_positions`
2. 透传到 `self.model()`

### Step 2: batch_core.py — batch_generate

1. `batch_generate(model, sampler, prompts, max_new_tokens, max_num_slots, ...)` 主循环：
   - 内部创建 `FCFSScheduler` + `BatchedKVCache`
   - `while scheduler.has_unfinished():`
   - finish done requests + free slots
   - admit waiting + prefill one by one
   - batched decode one step
   - sample + update per-request state

### Step 3: engine/__init__.py — 导出

1. 加 `batch_generate` 到 `__all__`

### Step 4: 测试

1. 10 个 L0 测试

## 算法核心

### 0. 调度策略边界

M3 的 BatchEngine 是 server-style continuous batching 的最小同步形态：持续维护 waiting/running/finished 集合，在 iteration 边界释放 finished 请求、admit waiting 请求，并对 running 集合执行 batched decode。

这里的 server-style 不等于 prefill-first，也不代表生产框架都采用 prefill-first。M3 不实现 vLLM V1 这类 decode-first + chunked prefill + token-budget mixed scheduling；M3 只做 fixed-slot continuous batching 的语义闭环。

关键不变量：

```text
waiting 请求不占 KV slot
running 请求才占 KV slot
finished / cancelled 请求必须释放 slot
```

### 1. Admit 请求并做 prefill

M3 先逐条 prefill：

```python
admitted = scheduler.admit_until_full()
for req in admitted:
    slot = cache.allocate_slot(req.request_id)
    req.slot_id = slot

    prompt_len = req.prompt_ids.shape[1]
    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
    logits = model(
        req.prompt_ids,
        position_ids=position_ids,
        kv_cache=cache,
        cache_slots=torch.tensor([slot]),
    )

    req.seq_len = prompt_len
    req.last_token = sampler(logits[:, -1, :])
    req.generated_tokens.append(req.last_token)
    req.num_generated = 1
    cache.seq_lens[slot] = prompt_len
```

### 2. 每轮 decode 重新组 batch

```python
running = list(scheduler.running.values())
cache_slots = torch.tensor([r.slot_id for r in running])
cache_positions = torch.tensor([r.seq_len for r in running])
next_tokens = torch.cat([r.last_token for r in running], dim=0)

logits = model(
    next_tokens,
    position_ids=cache_positions[:, None],
    kv_cache=cache,
    cache_slots=cache_slots,
    cache_positions=cache_positions,
)
```

### 3. 更新请求状态

```python
sampled = sampler(logits[:, -1, :])
for req, tok in zip(running, sampled):
    req.generated_tokens.append(tok)
    req.last_token = tok
    req.seq_len += 1
    req.num_generated += 1
    cache.seq_lens[req.slot_id] = req.seq_len

    if is_finished(req, tok):
        scheduler.mark_finished(req)
        cache.free_slot(req.request_id)
```

### 4. continuous batching 入口点

每轮 decode 结束后立刻尝试释放 finished 请求，并在后续 iteration 边界 admit waiting 请求：

```python
finish_done_requests_and_free_slots()
admit_waiting_requests_with_prefill()
```

这就是 iteration-level scheduling：finished 请求离开，新请求在下一个 iteration boundary 进入。注意 M3 不做 batching window，也不实现生产级 token-budget scheduler。

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|---|---|---|
| 1 | 单请求 batch_generate 等价 M2 generate | token 序列一致 | 精确或 sampler 固定 |
| 2 | 多请求输出数量 | 等于输入请求数 | 精确 |
| 3 | `max_num_slots=2` 时 running 不超过 2 | scheduler/cache invariant | 精确 |
| 4 | 短请求完成后释放 slot | slot 可被等待请求复用 | 精确 |
| 5 | 每轮 batch 重新形成 | batch size trace 符合预期 | 精确 |
| 6 | EOS 请求提前退出 | 不继续 decode 空步 | 精确 |
| 7 | max_new_tokens 到达即 finished | 输出长度正确 | 精确 |
| 8 | waiting queue 最终清空 | 所有请求 finished | 精确 |
| 9 | waiting 不占 KV slot | submit 后未 allocate slot | 精确 |
| 10 | finished 后下一轮可 admit | 释放 slot 后 pending 请求进入 | 精确 |

## DoD

- [ ] `batch_generate` 能提交并执行多个请求。
- [ ] prefill 逐条执行，decode 按 running batch 执行。
- [ ] 每轮 decode 结束后支持 finished 离开、waiting 进入。
- [ ] waiting 请求不占 KV slot，running 请求才占 slot。
- [ ] 不实现 token-budget scheduler / decode-first / mixed prefill-decode。
- [ ] EOS / max_new_tokens 完成条件正确。
- [ ] 单请求路径不破坏 M2 generate。
- [ ] `uv run pytest tests/unit/test_batch_engine.py -q` 通过。
- [ ] 相关 M1/M2 单测全绿。
- [ ] commit `feat(engine): add minimal continuous batching engine (M3-T4 done)`。

## 坑（按概率排序）

1. **prefill 后 seq_len off-by-one**：prefill 写入 prompt KV 后，decode 当前 token 位置应是 `prompt_len`。
2. **第一次 sampled token 的处理**：prefill 后采出的第一个 token 是否计入 output，要和 M2 generate 对齐。
3. **finished 请求继续参与下一轮 batch**：会造成空步和错误 slot 访问。
4. **释放 slot 后 req.slot_id 未清**：后续 debug 容易混淆。
5. **把 admit 放到固定 batch 结束后**：会退化成 static batching。

## 完成总结

### 最终实现

`batch_generate()` 纯函数，内部创建 `FCFSScheduler` + `BatchedKVCache`，主循环结构：

1. **admit + prefill**：`admit_until_full()` 返回新请求，逐条 prefill（B=1 forward）
2. **batched decode**：所有 running 请求组 batch，并行一步 decode
3. **update + finish**：采样、更新状态、检查 max_new_tokens / EOS，finished 释放 slot

### 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 纯函数 vs 类 | 纯函数 | 简单，和 M2 generate 对称 |
| seq_len 语义 | next write position | 和 M2 cur_len 对齐，和 nano-vllm 一致 |
| 复用 EngineCore | 不复用 | step() 不支持 kv_cache/cache_slots，直接持有 model+sampler |
| prefill batching | 不做 | 教学范围限制，留 M4/M5 |

### 修改文件

| 文件 | 改动 |
|---|---|
| `engine/batch_core.py` | 新建 `batch_generate()` 函数 |
| `engine/protocol.py` | `LLMModel` 协议加 `cache_slots`/`cache_positions` 参数 |
| `test_batch_engine.py` | 10 个 L0 测试 |

### 测试覆盖

10/10 通过，全量回归 176/176 通过。
