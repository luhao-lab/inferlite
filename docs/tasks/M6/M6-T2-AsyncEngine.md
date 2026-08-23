# M6-T2 AsyncEngine 常驻引擎

## 元信息
- **任务 ID**: T2
- **里程碑**: M6
- **状态**: ⬜ pending
- **前置**: T1
- **估时**: 5h

## 目标

**要解决什么问题**：
`batch_generate_loop()` 是同步函数——一次性收齐所有 prompt，跑完返回。HTTP 服务需要请求随时到达、引擎常驻运行、token 实时推送。

**做完是什么效果**：
```python
engine = AsyncEngine(model, tokenizer, config, ...)
await engine.start()

# 请求随时到达
req_id = await engine.submit(prompt_ids, params, max_tokens)

# 流式获取 token
async for token_id in engine.stream(req_id):
    print(tokenizer.decode([token_id]), end="", flush=True)

await engine.shutdown()
```

**不做什么**（边界）：
- 不做多进程（单进程 asyncio）
- 不做 IPC（同进程 asyncio.Queue）
- 不做复杂调度策略（复用 FCFSScheduler）
- 不做请求优先级/抢占

**在推理链路中的位置**：
```
HTTP request
  ↓
AsyncEngine.submit() → scheduler.submit(RequestState)
  ↓
AsyncEngine._run_loop()  [常驻 while True]
  ↓ _step()
admit → prefill → decode → sample
  ↓
token → asyncio.Queue → stream consumer → SSE
```

## 产出文件
- `inferlite/engine/async_engine.py` — AsyncEngine 类
- `inferlite/engine/__init__.py` — 导出
- `tests/unit/test_async_engine.py` — 单元测试

## 算法核心

```python
class AsyncEngine:
    """常驻异步推理引擎（vLLM V1 AsyncLLM 单进程简化版）。

    核心设计：
    - 常驻循环（_run_loop）在后台 asyncio.Task 中运行
    - model forward 是同步的，放 asyncio.to_thread 不阻塞事件循环
    - 每个请求一个 asyncio.Queue，token 采完立即推入
    - submit 可随时调用，不要求一次性收齐 prompt
    """

    def __init__(self, model, tokenizer, config,
                 max_num_seqs: int = 4,
                 num_blocks: int = 128,
                 block_size: int = 16,
                 device="cpu",
                 dtype=torch.float32):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # 复用 M4/M5 的调度器 + 分页缓存
        self.scheduler = FCFSScheduler(max_num_seqs=max_num_seqs)
        paged_cache = PagedKVCache.from_config(config, num_blocks, block_size, dtype, device)
        self.adapter = PagedCacheAdapter(paged_cache)
        self.adapter.bind_kv_cache(model)

        # 流式输出：每个请求一个 Queue
        self._stream_queues: dict[str, asyncio.Queue] = {}
        # 非流式：收集完整输出
        self._request_params: dict[str, SamplingParams] = {}
        self._request_generated: dict[str, list[int]] = {}

        self._running = False
        self._loop_task: asyncio.Task | None = None

    async def submit(self, prompt_ids: torch.Tensor,
                     params: SamplingParams,
                     max_new_tokens: int = 128,
                     eos_token_id: int | None = None) -> str:
        """提交请求，返回 request_id。"""
        request_id = str(uuid.uuid4())
        req = RequestState(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        self.scheduler.submit(req)
        self._stream_queues[request_id] = asyncio.Queue()
        self._request_params[request_id] = params
        self._request_generated[request_id] = []
        return request_id

    async def stream(self, request_id: str) -> AsyncIterator[int]:
        """流式获取生成的 token id。"""
        queue = self._stream_queues[request_id]
        while True:
            item = await queue.get()
            if item is None:  # 结束信号
                break
            yield item
        # 清理
        del self._stream_queues[request_id]
        del self._request_params[request_id]

    async def get_output(self, request_id: str) -> list[int]:
        """非流式：等待全部生成完成，返回 token id 列表。"""
        tokens = []
        async for token_id in self.stream(request_id):
            tokens.append(token_id)
        return tokens

    async def start(self):
        """启动常驻循环。"""
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())

    async def shutdown(self):
        """停止循环。"""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self):
        """常驻事件循环。"""
        while self._running:
            has_work = await asyncio.to_thread(self._step)
            if not has_work:
                await asyncio.sleep(0.001)  # 无请求时降低 CPU
            else:
                await asyncio.sleep(0)  # 有工作时立即让出事件循环

    def _step(self) -> bool:
        """执行一步 prefill/decode（同步，在线程池运行）。

        复用 loop.py 的 admit → prefill → decode 逻辑，
        但每采完一个 token 推入对应 asyncio.Queue。

        Returns True if there was work to do.
        """
        # 1. admit（从 waiting 取请求到 running）
        # 2. prefill（新请求的前向 + 首 token 采样）
        # 3. decode（running 请求的批量解码）
        # 4. 每个 token → asyncio.Queue.put_nowait()
        # 5. 完成请求 → Queue.put(None) + free cache
        ...
```

### _step() 与 loop.py 的关系

```text
batch_generate_loop（现有）：           AsyncEngine._step（新增）：

while has_unfinished():                每次调用只执行一步：
  admit + prefill                        admit + prefill（如有）
  decode                                  decode（如有）
  全部跑完才返回                            返回 True/False
                                           token → Queue
```

_step 从 loop.py 抽取核心逻辑，但不依赖 loop.py 的 while 循环。

### 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| forward 在哪运行 | `asyncio.to_thread` | model forward 是同步的，放线程池 |
| Queue 类型 | `asyncio.Queue` | 同进程，不需要 IPC |
| 结束信号 | `None` sentinel | 简单可靠 |
| 无请求时 | `sleep(0.001)` | 避免空转 CPU |
| seed 管理 | 每请求独立 SamplingProcessor | 不同请求的 seed 互不干扰 |

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|--------|-------------|------|
| 1 | submit + stream 基本功能 | 能拿到 token 流 | exact |
| 2 | 多请求并发 submit | 各自 stream 独立 | exact |
| 3 | 动态提交（运行中 submit 新请求） | 新请求被 admit | exact |
| 4 | 非流式 get_output | 返回完整 token 序列 | exact |
| 5 | shutdown 清理 | queue drain，task 结束 | exact |
| 6 | EOS 提前停止 | 遇到 EOS 发 None | exact |
| 7 | 与 batch_generate_paged 等价 | 同 prompt 同 seed 输出一致 | exact |
| 8 | 空引擎 idle | _step 返回 False，不报错 | no crash |

## DoD
- [ ] 测试 8/8 全绿
- [ ] commit `feat(engine): add AsyncEngine with streaming support (T2 done)`
- [ ] `engine/__init__.py` 导出 AsyncEngine
- [ ] PROGRESS.md 更新

## 坑（按概率排序）
1. `asyncio.to_thread` + `torch.no_grad()` + MPS 的交互
2. Queue put_nowait vs await put 的选择（_step 在线程池，要用 put_nowait + loop.call_soon_threadsafe）
3. 取消任务时 Queue 可能还有未消费数据
4. _step 中 scheduler/adapter 不是线程安全的（需要 asyncio.Lock 保护）
