# inferlite M6：API + SSE 服务化 完整设计

| 字段 | 内容 |
|---|---|
| 状态 | ✅ 完成（2026-08-23） |
| 前置 | M3 (Continuous Batching) + M4 (PagedAttention) + M5 (Prefix Cache) |
| 后续 | M7 MoE 教学版 |
| 测试 | 31 新增（15 sampling + 16 API），344 全量回归通过 |
| 主层 | L4 Server |

---

## 摘要

M5 完成了 prefix cache + continuous batching + paged attention 的完整推理引擎，但只能通过 Python API 调用（写代码才能用）。M6 的目标是把这个引擎包装成 OpenAI-compatible HTTP 服务：`inferlite serve` 一行命令启动，curl / OpenAI SDK / ChatBox 等客户端直接调用。

**M6 的核心收获：AsyncEngine 线程模型 + SSE 流式协议 + 采样流水线 + Pydantic schema 对齐。**

---

## 符号说明

| 符号 | 含义 | M6 典型值 |
|---|---|---|
| `request_id` | 每个推理请求的唯一 UUID | `str(uuid4())` |
| `output_queue` | per-request 的 token 输出队列（`queue.Queue`） | 每请求一个 |
| `_Sentinel` | 放入 output queue 的结束信号 | `finish_reason="stop"/"length"` |
| `_PendingRequest` | submit() 放入 pending queue 的请求包装 | 含 RequestState + queue + SamplingProcessor |
| `B` | 当前 decode batch 大小（= running 请求数） | 1~max_num_seqs |
| `V` | vocab_size（Qwen3-0.6B = 151936） | 151936 |
| `T` | temperature 采样温度 | 0.0~2.0 |
| `k` | top-k 过滤保留数 | -1（不过滤）或 1~V |
| `p` | top-p nucleus 阈值 | 0.0~1.0 |
| `penalty` | repetition penalty 系数 | 1.0~2.0 |
| SSE | Server-Sent Events 协议 | `data: {json}\n\n` |

---

## 1. M5 → M6 的关键变化

| 维度 | M5（Python API） | M6（HTTP 服务） |
|---|---|---|
| 调用方式 | `batch_generate_paged(model, sampler, prompts, ...)` | `POST /v1/chat/completions` |
| 请求模型 | 一次性收齐 prompts 列表 | 动态接收，随时 submit |
| 输出方式 | 跑完返回完整字符串列表 | 逐 token 推送（SSE）或等待完整响应 |
| 采样策略 | `GreedySampler`（argmax only） | `SamplingProcessor`（temperature/top-k/top-p/penalty） |
| 运行模式 | 函数调用，跑完退出 | 常驻 HTTP 服务（uvicorn） |
| 线程模型 | 单线程 | Main Thread (FastAPI) + Background Thread (Engine) |
| 协议格式 | Python dict | OpenAI-compatible JSON（Pydantic 校验） |

---

## 2. 与 vLLM V1 的简化对照

| 维度 | inferlite M6 | vLLM V1 |
|---|---|---|
| 进程模型 | 单进程（FastAPI + 后台线程） | 多进程（API Server + EngineCore + GPU Workers） |
| 通信方式 | `queue.Queue`（线程安全） | `multiprocessing.Queue` + `asyncio.Queue` |
| Engine 载体 | `threading.Thread` 后台线程 | `EngineCore` 独立进程 |
| 调度策略 | 复用 M3 FCFS | Token-budget scheduling |
| Chunked prefill | 不做（M10） | 支持 |
| CUDA Graph | 不做（M9） | 支持 |
| FlashAttention | 不做（M9） | 支持 |
| 多 GPU | 不做（M12+） | TP / PP |

一句话：vLLM 是生产级多进程分布式推理引擎，inferlite M6 是单进程教学版，核心价值在**理解原理**而非**追求性能**。

---

## 2.5 M6 如何与 M5 引擎连接

M6 没有重写 M5 的任何组件——它只是**把 M5 的组件组装到了一个后台线程里**，然后用 HTTP 接口暴露出来。M5 的 PagedKVCache、PagedCacheAdapter、FCFSScheduler、prefix caching 全部原样复用，一行都没改。M6 新增的只是**线程模型 + queue 通信 + HTTP 协议适配**这三层胶水。

### 全链路：从 curl 到 M5 引擎

```
curl POST /v1/chat/completions {"messages":[...], "stream":true}
  │
  ▼
┌─ cli.py ──────────────────────────────────────────────────────┐
│  加载 model + tokenizer → 创建 AsyncEngine → engine.start()   │
│  → create_app(engine) → uvicorn.run(app)                      │
└───────────────────────────────────────────────────────────────┘
  │
  ▼
┌─ app.py: chat_completions() ──────────────────────────────────┐
│  ① messages → tokenizer.apply_chat_template → prompt_ids      │
│  ② request 参数 → SamplingParams → SamplingProcessor          │
│  ③ engine.submit(prompt_ids, sampling_params)                  │
│     → 打包为 _PendingRequest → 放入 pending_queue             │
│     → 返回 (request_id, output_queue)                          │
│  ④ return StreamingResponse(_stream_response(...))             │
└───────────────────────────────────────────────────────────────┘
  │ pending_queue                          ▲ output_queue
  ▼                                        │
┌─ async_engine.py: _background_loop() ────┴──────────────────┐
│                                                              │
│  初始化 M5 组件（只在后台线程启动时执行一次）：                  │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ paged_cache  = PagedKVCache.from_config(...)   ← M4 │     │
│  │ adapter      = PagedCacheAdapter(paged_cache)  ← M4 │     │
│  │ adapter.bind_kv_cache(model)                        │     │
│  │ scheduler    = FCFSScheduler(max_num_seqs=...)  ← M3 │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  while running:                                              │
│    ① pending_queue.get() → scheduler.submit()     ← M3       │
│    ② adapter.can_admit()                          ← M4       │
│    ③ adapter.can_admit_with_cache()                ← M5       │
│    ④ adapter.allocate_with_cache()                 ← M5       │
│    ⑤ model(input_ids) — batched prefill            ← M1      │
│    ⑥ model(next_tokens) — batched decode            ← M2/M3  │
│    ⑦ adapter.cache.hash_blocks()                   ← M5       │
│    ⑧ output_queue.put(token_id) → 推给 HTTP 层               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
  │ output_queue
  ▼
┌─ app.py: _stream_response() ─────────────────────────────────┐
│  out_q.get() → tokenizer.decode → think_parser.feed()         │
│  → yield f"data: {chunk_json}\n\n"  → SSE 推给客户端          │
└───────────────────────────────────────────────────────────────┘
```

### 关键连接点：只有 3 处

**连接点 1：启动时组装 M5 组件**（`_background_loop()` 开头）

```python
# async_engine.py:210-228
paged_cache = PagedKVCache.from_config(...)    # M4
adapter = PagedCacheAdapter(paged_cache)       # M4
adapter.bind_kv_cache(self.model)
scheduler = FCFSScheduler(max_num_seqs=...)    # M3
```

和 M5 的 `batch_generate_paged()` 函数的初始化逻辑完全一样——只是从一次性函数调用变成了后台线程的常驻循环。

**连接点 2：请求进入**（HTTP → queue → scheduler）

```python
# app.py:97 — HTTP handler 调用
request_id, out_q = eng.submit(prompt_ids=prompt_ids, ...)

# async_engine.py:185 — submit() 只放入 pending queue
self._pending_queue.put(_PendingRequest(req_state, out_q, processor))

# async_engine.py:237-238 — 后台线程取出，交给 M3 scheduler
pending = self._pending_queue.get_nowait()
scheduler.submit(pending.request_state)
```

**连接点 3：推理执行**（复用 M4/M5 的 admit + cache 逻辑）

```python
# async_engine.py:258-275
while scheduler.waiting:
    req = scheduler.waiting[0]
    if not adapter.can_admit(prompt_len):          # M4: 空间检查
        break
    num_cached = adapter.can_admit_with_cache(...)  # M5: prefix cache 命中
    if num_cached > 0:
        adapter.allocate_with_cache(...)            # M5: cache-aware 分配
    else:
        adapter.allocate(req.request_id, prompt_len) # M4: 普通分配
```

### 扩展参数走三层链路

新增 API 参数（如 `min_tokens`、`frequency_penalty`）时，按影响范围走不同路径：

| 参数影响的层面 | 走哪条路径 | 举例 |
|---|---|---|
| **采样行为** | Schema → `SamplingParams` → `SamplingProcessor` | `temperature`, `top_k`, `frequency_penalty` |
| **生成控制** | Schema → `RequestState` → `_background_loop` | `max_tokens`, `min_tokens`, `stop` |
| **响应格式** | Schema → Handler 逻辑分支 | `stream`, `logprobs` |

每层只需加 1~2 行代码：L1 Schema 加字段声明 + Pydantic 校验，L2 Handler 取参数传给 engine，L3 Engine 在推理逻辑中使用。

---

## 3. AsyncEngine：线程模型

### 3.1 为什么需要后台线程

核心矛盾：**FastAPI 跑在 asyncio event loop 上，不能被阻塞**。但模型 forward 是 CPU/GPU 密集型的同步计算，如果在 async handler 里直接调 `model(input_ids)`，整个 event loop 会卡住，所有 HTTP 连接都无响应。

解法：两个线程，queue 通信。

```
┌─ Main Thread (uvicorn + asyncio) ─────┐
│                                       │
│  POST /v1/chat/completions            │
│    → engine.submit(prompt_ids)         │
│    → return (request_id, output_queue) │
│    → async for token in output_queue:  │
│        yield SSE chunk                 │
│                                       │
└───────────────┬───────────────────────┘
                │  queue.Queue (thread-safe)
                │
┌───────────────▼───────────────────────┐
│  Background Thread (Engine Loop)      │
│                                       │
│  while running:                       │
│    1. admit 新请求（分配 KV cache）     │
│    2. batched prefill（新请求）        │
│    3. batched decode（所有 running）   │
│    4. emit token → output_queue       │
│    5. 完成检查 → emit _Sentinel       │
│                                       │
└───────────────────────────────────────┘
```

### 3.2 GIL 不是瓶颈

Python GIL 通常被认为是多线程的性能杀手，但在这里不是问题：

- **model forward 释放 GIL**：PyTorch 的 C++ 后端在执行 tensor 运算时会释放 GIL
- **FastAPI 不抢 GIL**：async handler 大部分时间在 `await asyncio.sleep()` 等待 queue
- **queue.Queue 操作极快**：`put()` / `get()` 本身只持 GIL 微秒级

### 3.3 请求生命周期

```
HTTP handler 调用 engine.submit()
  → 创建 RequestState + SamplingProcessor + output_queue
  → 打包为 _PendingRequest 放入 self._pending_queue
  → 返回 (request_id, output_queue)

后台线程 _background_loop() 每轮：
  1. 从 _pending_queue 取出 → scheduler.submit() → 进入 waiting 队列
  2. waiting → admit（分配 KV cache blocks）→ 进入 running
  3. batched prefill → 首个 token → _emit_token(request_id, token_id)
  4. batched decode → 后续 token → _emit_token(request_id, token_id)
  5. 到达 max_tokens 或 EOS → _emit_sentinel(request_id, "stop"/"length")
  6. scheduler.mark_finished() + adapter.free() → 释放资源

HTTP handler 从 output_queue 读取：
  → 收到 int → decode 为文本 → yield SSE chunk
  → 收到 _Sentinel → yield 结束 chunk → yield [DONE]
  → cleanup_request(request_id) → 清理 output_queue
```

### 3.4 per-request 独立采样参数

每个 HTTP 请求可以指定不同的 `temperature` / `top_k` / `top_p` / `repetition_penalty`。实现方式：

- `submit()` 时为每个请求创建独立的 `SamplingProcessor` 实例
- 存入 `active_processors: dict[str, SamplingProcessor]`（request_id → processor）
- 后台线程的 decode 阶段，按 request_id 取出对应的 processor 做采样
- 请求完成时从 dict 中 pop 掉

### 3.5 源码走读：submit()

```python
# inferlite/engine/async_engine.py

def submit(self, prompt_ids, max_new_tokens=256, eos_token_id=None,
           sampling_params=None) -> tuple[str, queue.Queue]:
    request_id = str(uuid.uuid4())

    # 1. 构造 RequestState（与 batch_generate 的 scheduler.submit 等价）
    req_state = RequestState(
        request_id=request_id, prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
    )

    # 2. 构造 per-request 采样器（每个请求可以有独立的 temperature/top_k/top_p）
    if sampling_params is None:
        sampling_params = SamplingParams()
    processor = SamplingProcessor(sampling_params)

    # 3. 创建 output queue 并注册到 _output_queues dict
    out_q = queue.Queue()
    with self._queues_lock:
        self._output_queues[request_id] = out_q

    # 4. 打包为 _PendingRequest 放入 pending queue
    #    后台线程会在下一轮 _background_loop 中取出并 admit
    self._pending_queue.put(
        _PendingRequest(req_state, out_q, processor)
    )

    return request_id, out_q
```

关键点：`submit()` 是**非阻塞**的——它只把请求放入 queue 就返回，不等推理完成。HTTP handler 拿到 `(request_id, out_q)` 后自己决定是流式读还是等待完整结果。

### 3.6 源码走读：_background_loop() 核心片段

```python
# inferlite/engine/async_engine.py — _background_loop() 简化版

while self._running:
    # ── 1. 接收新请求 ──
    while not self._pending_queue.empty():
        pending = self._pending_queue.get_nowait()
        scheduler.submit(pending.request_state)
        active_processors[pending.request_state.request_id] = (
            pending.sampling_processor
        )

    # ── 2. 没有活跃请求就 sleep 1ms 避免 CPU 空转 ──
    if not scheduler.has_unfinished():
        time.sleep(0.001)
        continue

    # ── 3. Admit + Allocate ──
    admitted = []
    while scheduler.waiting:
        req = scheduler.waiting[0]
        if not adapter.can_admit(prompt_len):
            break
        num_cached = adapter.can_admit_with_cache(prompt_ids_list)
        if num_cached == -1:   # prefix cache 正在计算中，本轮跳过
            break
        scheduler.waiting.popleft()
        req.status = RequestStatus.RUNNING
        scheduler.running[req.request_id] = req
        admitted.append(req)
        # 分配 KV cache blocks（有 prefix 命中时只分配剩余部分）
        if num_cached > 0:
            adapter.allocate_with_cache(req.request_id, prompt_ids_list, num_cached)
        else:
            adapter.allocate(req.request_id, prompt_len)

    # ── 4. Batched Prefill（新请求的第一个 token）──
    if admitted:
        input_ids, positions = _build_prefill_batch(admitted, self.device)
        metadata = adapter.make_prefill_metadata(input_ids, positions, ...)
        with set_forward_context(metadata):
            logits = self.model(input_ids, positions=positions)
        for i, req in enumerate(admitted):
            proc = active_processors.get(req.request_id)
            req_logits = logits[i, plen - 1, :].unsqueeze(0)  # [1, V]
            token = proc(req_logits)       # 用该请求的采样器采样首个 token
            self._emit_token(req.request_id, token.item())

    # ── 5. Batched Decode（所有 running 请求并行一步）──
    running = list(scheduler.running.values())
    if not running:
        continue
    next_tokens, positions = _build_decode_batch(running, self.device)
    metadata = adapter.make_decode_metadata(next_tokens, positions)
    with set_forward_context(metadata):
        logits = self.model(next_tokens, positions=positions)
    for idx, req in enumerate(running):
        proc = active_processors.get(req.request_id)
        gen_ids = [t.item() for t in req.generated_tokens]
        proc.set_generated_ids([gen_ids])  # 注入已生成 token（for repetition penalty）
        token = proc(req_logit)
        # 完成检查
        is_done = (req.num_generated >= req.max_new_tokens
                   or token_id == eos_token_id)
        if is_done:
            self._emit_sentinel(req.request_id, "stop" or "length")
        else:
            self._emit_token(req.request_id, token_id)

    # ── 6. 清理完成的请求 ──
    for req in finished_reqs:
        scheduler.mark_finished(req)
        adapter.free(req.request_id)       # 释放 KV cache blocks
        active_processors.pop(req.request_id, None)  # 清理采样器
```

`_emit_token()` 和 `_emit_sentinel()` 的实现极简——就是往对应 request 的 output queue 里 `put()` 一个值：

```python
def _emit_token(self, request_id: str, token_id: int) -> None:
    with self._queues_lock:
        q = self._output_queues.get(request_id)
    if q is not None:
        q.put(token_id)

def _emit_sentinel(self, request_id: str, finish_reason: str) -> None:
    with self._queues_lock:
        q = self._output_queues.get(request_id)
    if q is not None:
        q.put(_Sentinel(finish_reason=finish_reason))
```

---

## 4. SSE 流式协议

### 4.1 通俗理解：HTTP vs SSE

**HTTP = 餐厅点菜**：你跟服务员说"我要宫保鸡丁"（发请求），服务员去后厨做完，一次性端上来（返回响应），吃完走人（连接断开）。特点是**一问一答**。

对应 inferlite 的非流式模式（`stream: false`）：客户端发一个请求，服务端等全部 token 生成完，一次性返回完整 JSON。问题：如果生成 500 个 token，客户端屏幕空白等好几秒。

**SSE = 火锅店传送带**：坐下来之后，传送带**持续不断地**把菜送过来——做好一盘送一盤，不用等全部做完。特点是**一次坐下，持续收货**。

对应 inferlite 的流式模式（`stream: true`）：每生成一个 token 就立刻推给客户端，用户看到字一个个蹦出来（ChatGPT 的逐字输出效果就是这么做的）。

**为什么不用轮询（polling）**：每 100ms 问一次"生成了吗？"→ 大量无效请求浪费带宽，延迟高。SSE 是服务器**主动推**，有了就发，没有就等，零浪费。

**为什么不用 WebSocket**：WebSocket 需要握手升级协议、支持双向通信、断线重连要自己实现。LLM 流式输出只需要"服务器 → 客户端"单向推送，SSE（普通 HTTP 长连接）就够了，更简单。

| | HTTP 一问一答（非流式） | SSE 持续推送（流式） |
|---|---|---|
| 比喻 | 餐厅点菜，菜做好一起上 | 火锅传送带，做好一盘送一盤 |
| 用户体验 | 等很久，然后突然全出来 | 立刻开始看到内容，逐字蹦出来 |
| 连接 | 一个请求 → 一个响应 → 断开 | 一个请求 → 多个数据帧 → 断开 |
| 格式 | 返回一个完整 JSON | 每帧 `data: {json}\n\n` |
| 结束信号 | HTTP 响应自然结束 | `data: [DONE]\n\n` |
| inferlite 对应 | `stream: false` | `stream: true` |

### 4.2 OpenAI ChatCompletionStream 格式

一个完整的流式响应包含以下 chunk 序列：

```
# 1. 首 chunk：告知客户端角色
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant","content":""}}]}

# 2. 中间 chunks：逐 token 推送
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"你"}}]}
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"好"}}]}
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{"content":"！"}}]}

# 3. 末 chunk：告知结束原因
data: {"id":"chatcmpl-abc","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

# 4. 终止信号
data: [DONE]
```

关键规则：
- 每个 chunk 是 `data: ` + JSON + `\n\n`（两个换行分隔）
- `[DONE]` 不是 JSON，是纯文本终止信号
- `finish_reason` 取值：`"stop"`（遇到 EOS）/ `"length"`（到达 max_tokens）

### 4.3 FastAPI StreamingResponse

```python
from fastapi.responses import StreamingResponse

return StreamingResponse(
    _stream_response(request_id, out_q, engine, model_name),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",       # 不缓存 SSE
        "X-Accel-Buffering": "no",         # nginx 不缓冲
    },
)
```

`_stream_response` 是一个 async generator：每 `yield` 一个字符串就推一帧 SSE 给客户端。底层由 uvicorn 的 ASGI server 负责 chunked transfer encoding。

### 4.4 在 asyncio 中读 thread-safe queue

`queue.Queue` 是线程安全的，但它的 `get()` 是阻塞调用——如果在 async handler 里直接 `await out_q.get()`，会阻塞 event loop。

解法：`get(timeout=0.05)` + `asyncio.sleep(0.001)` 轮询：

```python
while True:
    try:
        item = out_q.get(timeout=0.05)   # 最多等 50ms
    except queue.Empty:
        await asyncio.sleep(0.001)       # 让出 event loop
        continue

    if isinstance(item, _Sentinel):
        break
    # 处理 token_id ...
```

延迟开销：最坏情况下每个 token 多等 1ms（`asyncio.sleep` 的最小粒度），对于 LLM 推理来说可以忽略。

### 4.5 源码走读：_stream_response() 生成器

```python
# inferlite/server/app.py — _stream_response() 核心逻辑

async def _stream_response(request_id, out_q, engine, model_name):
    chunk_id = f"chatcmpl-{request_id[:8]}"
    created = int(time.time())

    # ① 首 chunk：告知客户端角色
    first_chunk = ChatCompletionStreamChunk(
        id=chunk_id, created=created, model=model_name,
        choices=[StreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant", content=""),
        )],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    # ② 逐 token 流式推送
    think_parser = _StreamingThinkParser()  # 状态机：检测 <think>...</think> 标签
    while True:
        try:
            item = out_q.get(timeout=0.05)  # 最多等 50ms
        except queue.Empty:
            await asyncio.sleep(0.001)     # 让出 event loop
            continue

        if isinstance(item, _Sentinel):    # 生成结束
            finish_reason = item.finish_reason
            break

        # 每收到 1 个 token 就 decode 一次（低延迟优先）
        text = engine.tokenizer.decode([item], skip_special_tokens=True)
        if text:
            # 通过状态机分流到 reasoning_content 或 content
            for field, delta in think_parser.feed(text):
                if field == "reasoning_content":
                    delta_msg = DeltaMessage(reasoning_content=delta)
                else:
                    delta_msg = DeltaMessage(content=delta)
                chunk = ChatCompletionStreamChunk(
                    id=chunk_id, created=created, model=model_name,
                    choices=[StreamChoice(index=0, delta=delta_msg)],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

    # ③ flush 思维链解析器剩余 buffer
    for field, delta in think_parser.flush():
        # ... 同上 yield chunk ...

    # ④ 结束 chunk：发送 finish_reason
    end_chunk = ChatCompletionStreamChunk(
        id=chunk_id, created=created, model=model_name,
        choices=[StreamChoice(
            index=0, delta=DeltaMessage(), finish_reason=finish_reason,
        )],
    )
    yield f"data: {end_chunk.model_dump_json()}\n\n"

    # ⑤ 终止信号
    yield "data: [DONE]\n\n"
```

整个生成器的结构就是 **首 chunk → 循环 yield content chunks → flush → 末 chunk → [DONE]**。每次 `yield` 的字符串就是一个 SSE 事件，由 FastAPI 的 `StreamingResponse` 通过 chunked transfer encoding 推给客户端。

---

## 5. 采样流水线

### 5.1 五步流水线

```
logits [1, V]（词表大小 151936 维的原始分数）
   │
   ▼ ① repetition penalty
     对已生成过的 token 做 logit 惩罚
     logit[v] > 0 → logit[v] / penalty
     logit[v] ≤ 0 → logit[v] * penalty
   │
   ▼ ② temperature scaling
     logits = logits / T
     T > 1 → 分布更平滑（更随机）
     T < 1 → 分布更锐利（更集中）
     T = 0 → 退化为 greedy（跳过后续步骤）
   │
   ▼ ③ top-k filtering
     只保留 logit 最高的 k 个 token，其余设为 -inf
   │
   ▼ ④ top-p (nucleus) filtering
     保留累积概率 ≤ p 的最小 token 集合
     动态候选集：概率集中时候选少，分散时候选多
   │
   ▼ ⑤ softmax → multinomial sampling
     从过滤后的概率分布中采样 1 个 token
```

### 5.2 每步的数学直觉

**Repetition penalty**（Ctrl et al., 2019）：
- 目的：降低模型重复生成相同 token 的概率
- 方法：对已出现 token 的 logit 做除法（正值）或乘法（负值），使其概率降低
- 关键：必须区分正负 logit 的方向，否则惩罚会反向

**Temperature scaling**：
- 本质是对 softmax 前的 logit 做缩放
- `T → 0`：softmax 退化为 one-hot（argmax），输出确定性最强
- `T → ∞`：softmax 退化为均匀分布，完全随机
- 实践中 T ∈ [0.5, 1.5] 效果最好

**Top-k filtering**：
- 简单粗暴：只保留 logit 最高的 k 个 token
- 问题：固定 k 值不灵活——有时只有 2 个合理选项，有时有 100 个

**Top-p (nucleus) filtering**（Holtzman et al., 2020）：
- 解决 top-k 的固定候选集问题
- 动态选择：累积概率达到 p 就停
- 例：`top_p=0.9` 表示只从占总概率 90% 的 token 中采样

### 5.3 实现细节：全部在 fp32 下操作

采样流水线中的所有操作（penalty / temperature / softmax / log）都在 fp32 下进行，即使模型权重是 bf16：

```python
logits = logits.float()   # bf16 → fp32
# ... 5 步采样 ...
# multinomial 直接输出 int64 token id，无需再转回
```

原因：softmax 中的 `exp()` 在 bf16 下容易溢出，`log()` 精度不足。

### 5.4 源码走读：SamplingProcessor.__call__()

```python
# inferlite/sampler/sampling.py — SamplingProcessor.__call__()

def __call__(self, logits: torch.Tensor) -> torch.Tensor:
    # logits [B, V] -> next_token_ids [B, 1]

    # ── greedy 快速路径 ──
    # temperature=0 时跳过所有采样步骤，直接 argmax（确定性输出）
    if self.params.temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 后续操作在 fp32 下做，避免 bf16/fp16 下 softmax / log 精度不足
    logits = logits.float()

    # ── ① repetition penalty ──
    # 对已生成过的 token 做 logit 惩罚：
    #   logit > 0 -> logit / penalty  (正值变小)
    #   logit <= 0 -> logit * penalty  (负值更负)
    if self.params.repetition_penalty != 1.0 and self._generated_ids is not None:
        for i, ids in enumerate(self._generated_ids):
            unique_ids = list(set(ids))
            prev_logits = logits[i, unique_ids]
            logits[i, unique_ids] = torch.where(
                prev_logits > 0,
                prev_logits / penalty,
                prev_logits * penalty,
            )

    # ── ② temperature scaling ──
    logits = logits / self.params.temperature

    # ── ③ top-k filtering ──
    # kthvalue 找第 k 小的值，比它小的全部 mask 为 -inf
    if self.params.top_k > 0:
        threshold = torch.kthvalue(
            logits, logits.size(-1) - top_k + 1, dim=-1
        ).values
        logits = logits.masked_fill(
            logits < threshold.unsqueeze(-1), float("-inf")
        )

    # ── ④ top-p (nucleus) filtering ──
    # 降序排列 -> softmax -> 累积概率 -> 找到超过 p 的位置 -> mask
    # 关键：用 cum_probs - sorted_probs > p 保留第一个超过阈值的 token
    if self.params.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cum_probs - sorted_probs > self.params.top_p
        # scatter mask 回原位置
        logits.scatter_(
            -1, sorted_indices,
            sorted_mask.float().mul(float("-inf"))
        )

    # ── ⑤ softmax -> multinomial sampling ──
    probs = torch.softmax(logits, dim=-1)
    next_tokens = torch.multinomial(
        probs, num_samples=1, generator=self._generator
    )
    return next_tokens  # [B, 1]
```

注意 `torch.multinomial` 的 `generator` 参数：如果 `SamplingParams.seed` 不为 None，使用固定种子的 `torch.Generator`，保证相同输入产生相同输出（可复现采样）。

---

## 6. OpenAI Schema 对齐

### 6.1 Pydantic 模型层次

```
ChatCompletionRequest          ← 请求体
├── messages: list[ChatMessage]
├── stream: bool
├── temperature / top_p / top_k / repetition_penalty / seed
└── max_tokens / stop / model

ChatCompletionResponse         ← 非流式响应
├── id / object / created / model
├── choices: list[ResponseChoice]
│   └── message: ResponseMessage (role + content + reasoning_content)
└── usage: UsageInfo

ChatCompletionStreamChunk      ← 流式 SSE chunk
├── id / object / created / model
└── choices: list[StreamChoice]
    └── delta: DeltaMessage (role? + content? + reasoning_content?)
```

### 6.2 Pydantic 校验

利用 Pydantic 的 `Field` 约束自动校验非法值：

```python
temperature: float = Field(default=0.0, ge=0.0, le=2.0)  # [0, 2]
top_p: float = Field(default=1.0, ge=0.0, le=1.0)        # [0, 1]
max_tokens: int = Field(default=256, ge=1)                 # ≥ 1
repetition_penalty: float = Field(default=1.0, ge=1.0)    # ≥ 1.0
```

客户端传入非法值时，FastAPI 自动返回 422 Validation Error，无需手动检查。

### 6.2.1 源码走读：Pydantic 模型定义

```python
# inferlite/server/schemas.py — 核心 Pydantic 模型

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="qwen3")
    messages: list[ChatMessage]           # [{role: "user", content: "Hi"}]
    stream: bool = Field(default=False)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1)        # -1 = 不过滤
    max_tokens: int = Field(default=256, ge=1)
    repetition_penalty: float = Field(default=1.0, ge=1.0)
    seed: int | None = Field(default=None)

class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str
    reasoning_content: str | None = None  # Qwen3 <think> 内容

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"       # 区别于 chunk 的 "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "qwen3"
    choices: list[ResponseChoice]
    usage: UsageInfo                      # 非流式才有

class DeltaMessage(BaseModel):
    role: str | None = None               # 首 chunk 设置
    content: str | None = None            # 中间 chunk 设置
    reasoning_content: str | None = None  # 流式思维链

class ChatCompletionStreamChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "qwen3"
    choices: list[StreamChoice]           # delta + finish_reason
```

`Field(default_factory=...)` 用于延迟计算默认值（如 `int(time.time())`），确保每次创建实例时获取当前时间而非模块加载时间。

### 6.3 请求字段到引擎层的映射

```
API 层（Pydantic）              →  引擎层
───────────────────────────────────────────────────
messages                         →  tokenizer.apply_chat_template() → prompt token ids
temperature / top_k / top_p /   →  SamplingParams → SamplingProcessor（per-request 实例）
  repetition_penalty / seed
max_tokens                       →  RequestState.max_new_tokens
stream                           →  StreamingResponse vs JSONResponse
```

---

## 7. 思维链映射

### 7.1 Qwen3 thinking 模式

Qwen3 在 thinking 模式下的输出格式：

```
<think>推理过程...（可能很长）</think>
回答内容
```

### 7.2 提取策略

用正则 `<think>(.*?)</think>` 提取标签内容，映射到 `reasoning_content` 字段：

| 情况 | content | reasoning_content |
|------|---------|-------------------|
| 无 `<think>` 标签 | 全部文本 | `None` |
| 完整 `<think>...</think>` | 标签外文本 | 标签内文本 |
| 未闭合 `<think>...` | `""` | 标签后全部文本 |

### 7.3 流式思维链分离

流式模式下，通过 `_StreamingThinkParser` 状态机实时检测 `<think>` / `</think>` 标签，将文本分流到 `reasoning_content` 和 `content`：

**状态转移**：
```
idle ──看到 <think>──▶ in_think ──看到 </think>──▶ after_think
  │                       │                            │
  ▼                       ▼                            ▼
输出为 content       输出为 reasoning_content       输出为 content
```

**标签边界处理**：tokenizer 可能把 `</think>` 拆成多个 token（如 `"</thin"` + `"k>"`），状态机通过 buffer 缓存尾部 `len(tag)-1` 个字符，等下一个 token 确认后再输出，避免误判。

**流式输出示例**：
```
data: {"choices":[{"delta":{"reasoning_content":"让我想想"}}]}
data: {"choices":[{"delta":{"reasoning_content":"...逐步分析"}}]}
data: {"choices":[{"delta":{"content":"答案是"}}]}        ← </think> 后切换为 content
data: {"choices":[{"delta":{"content":"42"}}]}
```

**未闭合标签**：如果生成结束时 `</think>` 还没出现，`flush()` 将剩余内容全部作为 `reasoning_content` 推送。

**测试覆盖**：7 个测试（`TestStreamingThinkParser`），覆盖无标签 / 完整标签 / 标签拆分 / 未闭合 / 单 token 含标签 / 空输入 / buffer 边界。

### 7.4 源码走读：非流式 _split_thinking()

```python
# inferlite/server/app.py

import re

def _split_thinking(text: str) -> tuple[str, str | None]:
    # 从完整文本中提取 <think>...</think> 内容
    # 返回 (content, reasoning_content)
    if "<think>" not in text:
        return text, None

    # re.DOTALL 使 . 匹配换行符（thinking 可能跨多行）
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = list(pattern.finditer(text))

    if not matches:
        # 有 <think> 但没有 </think>：全部作为 reasoning
        if "<think>" in text:
            start = text.index("<think>") + len("<think>")
            return "", text[start:]
        return text, None

    last = matches[-1]
    reasoning = last.group(1).strip()
    content = text[:last.start()] + text[last.end():]
    return content.strip(), reasoning
```

### 7.5 源码走读：流式 _StreamingThinkParser 状态机

```python
# inferlite/server/app.py — _StreamingThinkParser

_THINK_START = "<think>"    # 7 chars
_THINK_END = "</think>"    # 8 chars

class _StreamingThinkParser:
    def __init__(self) -> None:
        self._state = "idle"    # idle / in_think / after_think
        self._buffer = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        # 接收文本片段，返回 [(field, delta), ...] 列表
        self._buffer += text
        results = []

        while self._buffer:
            if self._state == "idle":
                idx = self._buffer.find(_THINK_START)
                if idx != -1:
                    # 标签前的文本作为 content
                    if idx > 0:
                        results.append(("content", self._buffer[:idx]))
                    # 跳过 <think> 标签本身
                    self._buffer = self._buffer[idx + len(_THINK_START):]
                    self._state = "in_think"
                else:
                    # 保留尾部 len(tag)-1 个字符（可能是不完整的标签前缀）
                    hold = len(_THINK_START) - 1   # 6 chars
                    safe = len(self._buffer) - hold
                    if safe > 0:
                        results.append(("content", self._buffer[:safe]))
                        self._buffer = self._buffer[safe:]
                    break  # 等更多文本

            elif self._state == "in_think":
                idx = self._buffer.find(_THINK_END)
                if idx != -1:
                    # </think> 前的文本作为 reasoning
                    if idx > 0:
                        results.append(
                            ("reasoning_content", self._buffer[:idx])
                        )
                    self._buffer = self._buffer[idx + len(_THINK_END):]
                    self._state = "after_think"
                else:
                    hold = len(_THINK_END) - 1     # 7 chars
                    safe = len(self._buffer) - hold
                    if safe > 0:
                        results.append(
                            ("reasoning_content", self._buffer[:safe])
                        )
                        self._buffer = self._buffer[safe:]
                    break

            elif self._state == "after_think":
                # 终态：所有内容都是 content
                results.append(("content", self._buffer))
                self._buffer = ""
                break

        return results

    def flush(self) -> list[tuple[str, str]]:
        # 流结束时 flush 剩余 buffer
        if not self._buffer:
            return []
        results = []
        if self._state == "in_think":
            results.append(("reasoning_content", self._buffer))
        elif self._state == "idle":
            results.append(("content", self._buffer))
        self._buffer = ""
        return results
```

**buffer 机制详解**：假设 tokenizer 输出了 `"<thi"` + `"nk>reasoning"`。第一次 `feed("<thi")` 时，`idle` 状态找不到 `<think>`，但尾部 `"<thi"` 可能是 `<think>` 的前缀（长度 4 < hold=6），所以全部缓存。第二次 `feed("nk>reasoning")` 时，buffer 变为 `"<think>reasoning"`，此时找到 `<think>` 标签，状态切换到 `in_think`，输出 `"reasoning"` 为 reasoning_content。

---

## 8. 文件清单

| 文件 | 类型 | 行数 | 职责 |
|------|------|------|------|
| `inferlite/sampler/sampling.py` | 新建 | ~150 | SamplingParams + SamplingProcessor |
| `inferlite/server/schemas.py` | 新建 | ~130 | Pydantic schemas（Request/Response/Chunk） |
| `inferlite/engine/async_engine.py` | 新建 | ~380 | AsyncEngine（后台线程 + per-request queue） |
| `inferlite/server/app.py` | 新建 | ~320 | FastAPI app + SSE + thinking 映射 |
| `inferlite/server/cli.py` | 新建 | ~160 | `inferlite serve` CLI 命令 |
| `inferlite/server/__init__.py` | 修改 | ~12 | 导出 create_app + AsyncEngine |
| `inferlite/engine/__init__.py` | 修改 | ~5 | 导出 AsyncEngine |
| `inferlite/sampler/__init__.py` | 修改 | ~15 | 导出 SamplingParams + SamplingProcessor |
| `tests/unit/test_sampling.py` | 新建 | ~190 | 15 个采样测试 |
| `tests/unit/test_openai_api.py` | 新建 | ~270 | 16 个 API/schema/SSE 测试 |

---

## 9. 测试覆盖

### 9.1 测试分类

| 测试类 | 数量 | 覆盖范围 |
|--------|------|----------|
| `TestGreedyPath` | 3 | temperature=0 → argmax 等价 |
| `TestTemperature` | 3 | 高温更随机、低温更集中 |
| `TestTopK` | 2 | top-k 过滤正确性 |
| `TestTopP` | 2 | nucleus sampling 正确性 |
| `TestRepetitionPenalty` | 2 | penalty 降低重复概率 |
| `TestSeedReproducibility` | 2 | seed 可复现 |
| `TestInterfaceCompatibility` | 1 | 接口兼容 GreedySampler |
| `TestChatCompletionRequest` | 4 | 请求 schema 校验 |
| `TestChatCompletionResponse` | 2 | 响应格式对齐 |
| `TestStreamChunk` | 4 | SSE chunk 格式 |
| `TestSplitThinking` | 3 | 思维链提取 |
| `TestFastAPIApp` | 2 | 服务启动 smoke test |
| **总计** | **31** | |

### 9.2 测试策略

- **采样测试**：全部使用手工构造的 logits（不需要真实模型），运行快且确定性高
- **统计性测试**：通过循环多次采样 + set 去重来验证分布特征（如"高温下应该采样到多个不同 token"）
- **API 测试**：schema 测试直接用 Pydantic 构造/校验；E2E 用 FastAPI TestClient + MagicMock engine
- **每个 test class 对应一个独立的验证维度**，方便定位问题

---

## 10. 已知局限性及后续路径

| 当前短板 | 根因 | 解决里程碑 |
|----------|------|------------|
| ~~流式不推送 reasoning_content~~ | ✅ 已支持（StreamingThinkParser 状态机） | — |
| ~~usage 统计 prompt_tokens=0~~ | ✅ 已修复（传入 prompt_len） | — |
| ~~MPS OOM 崩机~~ | ✅ 已修复（缺少 torch.no_grad() 导致计算图累积，见 L8） | — |
| 无 Web UI | CLI 足够 | 可选 Gradio/Streamlit |
| 单进程架构 | 教学简化 | M12+ 可扩展 |
| 无 chunked prefill | M3 FCFS 足够 | M10 |
| 无量化 / 多 GPU | 不在 M6 范围 | M12+ |
| 无 stop sequences | 预留字段未实现 | 可选扩展 |

---

## 11. 后续 M7 入口

M6 完成后，inferlite 具备完整的推理服务能力：

- ✅ M1：Qwen3 模型实现 + 数值对齐
- ✅ M2：KV Cache（单请求加速）
- ✅ M3：Continuous Batching（多请求并发）
- ✅ M4：PagedAttention（显存分页管理）
- ✅ M5：Prefix Cache（前缀复用）
- ✅ M6：API + SSE（HTTP 服务化）

**M7 候选方向**（详见 PLAN.md §5）：
- MoE 模型支持（Mixtral / DeepSeek-V2）
- 推测解码（Speculative Decoding）
- 长上下文（Chunked Prefill + YaRN）
- 性能优化（Triton kernel / FlashAttention）
