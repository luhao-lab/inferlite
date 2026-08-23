"""异步推理引擎：把 batch_generate_loop 改成常驻后台服务。

对齐 vLLM V1 的 AsyncLLM 架构（单进程简化版）：
  - 后台线程运行常驻的 engine loop（admit → prefill → decode → emit）
  - 每个请求一个 queue.Queue，token 生成后立即推入
  - FastAPI handler 从 queue 读取 token，组装 SSE chunk 推给客户端

线程模型：
  ┌─ Main Thread (FastAPI + uvicorn) ─┐
  │  HTTP handler → submit() → queue  │
  │  HTTP handler ← stream() ← queue  │
  └───────────────────────────────────┘
              ↕ queue.Queue (thread-safe)
  ┌─ Background Thread (Engine Loop) ─┐
  │  pending queue → admit → prefill  │
  │  decode step → emit tokens       │
  │  finish → emit _Sentinel         │
  └───────────────────────────────────┘

与 batch_generate_loop 的区别：
  - batch_generate_loop：一次性收齐 prompts，跑完返回完整结果
  - AsyncEngine：动态接收请求，逐 token 推送结果
"""

import queue
import threading

import torch

from inferlite.engine.context import set_forward_context
from inferlite.engine.loop import _build_decode_batch, _build_prefill_batch
from inferlite.scheduler.request import RequestState, RequestStatus


class _Sentinel:
    """放入 output queue 的信号，标记请求结束或错误。"""

    def __init__(self, finish_reason: str = "stop", error: str | None = None):
        self.finish_reason = finish_reason
        self.error = error


class _PendingRequest:
    """submit() 放入 pending queue 的请求包装。"""

    def __init__(
        self,
        request_state: RequestState,
        output_queue: queue.Queue,
        sampling_processor,
    ):
        self.request_state = request_state
        self.output_queue = output_queue
        self.sampling_processor = sampling_processor


class AsyncEngine:
    """异步推理引擎，参考 vLLM V1 AsyncLLM（单进程简化版）。

    架构：
      - 后台线程运行 _background_loop()，持续执行 admit → prefill → decode
      - submit() 将请求放入 pending queue（thread-safe）
      - stream() 从 per-request output queue 读取 token（async generator）

    使用方式：
        engine = AsyncEngine(model, tokenizer, config, device, dtype)
        engine.start()

        # 提交请求
        request_id, stream = engine.submit(prompt_ids, max_tokens, sampling_params)

        # 流式读取 token
        async for token_id in stream:
            yield tokenizer.decode([token_id])

        engine.stop()
    """

    def __init__(
        self,
        model,
        tokenizer,
        config,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        num_blocks: int = 128,
        block_size: int = 16,
    ) -> None:
        """初始化 AsyncEngine。

        Args:
            model: Qwen3ForCausalLM 实例（已加载权重）
            tokenizer: AutoTokenizer 实例
            config: ModelConfig（模型超参）
            device: 推理设备（cpu/mps/cuda）
            dtype: 模型和 KV cache 的 dtype
            num_blocks: PagedKVCache 的物理 block 总数
            block_size: 每个 block 的 token 数
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.dtype = dtype

        # ── 引擎核心组件（在 _background_loop 中初始化）──
        self._adapter = None
        self._scheduler = None

        # ── 线程通信 ──
        # pending queue：submit() → background thread
        self._pending_queue: queue.Queue[_PendingRequest] = queue.Queue()
        # per-request output queues：background thread → stream()
        # key = request_id，value = queue.Queue（token_id: int 或 _Sentinel）
        self._output_queues: dict[str, queue.Queue] = {}
        self._queues_lock = threading.Lock()

        # ── 后台线程控制 ──
        self._thread: threading.Thread | None = None
        self._running = False

        # ── Cache 参数 ──
        self._num_blocks = num_blocks
        self._block_size = block_size

    def start(self) -> None:
        """启动后台引擎线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止后台引擎线程，等待退出。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def submit(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        eos_token_id: int | None = None,
        sampling_params=None,
    ) -> tuple[str, "queue.Queue"]:
        """提交推理请求（线程安全，可从任意线程调用）。

        Args:
            prompt_ids: [1, prompt_len] 的 token id tensor
            max_new_tokens: 最大生成 token 数
            eos_token_id: EOS token id（None 表示不用 EOS 停止）
            sampling_params: SamplingParams 实例（None 表示 greedy）

        Returns:
            (request_id, output_queue) 元组。
            output_queue 中依次产出：int (token_id) → ... → _Sentinel (结束信号)。
        """
        import uuid

        from inferlite.sampler.sampling import SamplingParams, SamplingProcessor

        request_id = str(uuid.uuid4())

        # 构造 RequestState（与 batch_generate 的 scheduler.submit 等价）
        req_state = RequestState(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )

        # 构造 per-request 采样器（每个请求可以有独立的 temperature/top_k/top_p）
        if sampling_params is None:
            sampling_params = SamplingParams()
        processor = SamplingProcessor(sampling_params)

        # 创建 output queue 并注册
        out_q: queue.Queue = queue.Queue()
        with self._queues_lock:
            self._output_queues[request_id] = out_q

        # 放入 pending queue，后台线程会在下一轮 admit 时取出
        self._pending_queue.put(_PendingRequest(req_state, out_q, processor))

        return request_id, out_q

    # ── 后台引擎循环 ──

    def _background_loop(self) -> None:
        """后台线程的主循环：持续执行 admit → prefill → decode。

        在独立线程中运行，不阻塞 FastAPI 的 asyncio event loop。

        每轮循环的完整流程：
          1. Admit：从 pending queue 取出新请求，注册到 scheduler.waiting
          2. 空检查：没有活跃请求则 sleep 1ms 避免 CPU 空转
          3. Allocate：为 waiting 中的请求分配 KV cache blocks
          4. Prefill：对新 admitted 的请求做 batched prefill forward，产出首个 token
          5. Decode：对所有 running 请求做一步 batched decode forward，产出下一个 token
          6. 完成检查：到达 max_new_tokens 或遇到 EOS 的请求标记完成、释放资源

        注意：prefill 和 decode 在同一个循环步里执行，这意味着新请求的 prefill
        和老请求的 decode 共享同一次 model forward（通过不同的 attention metadata）。
        这里简化为先 prefill 再 decode（两次 forward），后续可合并为一次。
        """
        # 初始化 cache 和 adapter（必须在后台线程中做，因为 model 可能绑定了 GPU context）
        from inferlite.cache import PagedKVCache
        from inferlite.cache.adapter import PagedCacheAdapter
        from inferlite.scheduler.fcfs import FCFSScheduler

        paged_cache = PagedKVCache.from_config(
            config=self.config,
            num_blocks=self._num_blocks,
            block_size=self._block_size,
            dtype=self.dtype,
            device=self.device,
        )
        adapter = PagedCacheAdapter(paged_cache)
        adapter.bind_kv_cache(self.model)
        self._adapter = adapter

        # max_num_seqs = num_blocks（每个请求至少需要 1 个 block）
        scheduler = FCFSScheduler(max_num_seqs=self._num_blocks)
        self._scheduler = scheduler

        # 活跃的 per-request 采样器：request_id → SamplingProcessor
        active_processors: dict[str, object] = {}

        _is_mps = torch.backends.mps.is_available()

        try:
            self._run_engine_loop(scheduler, adapter, active_processors, _is_mps)
        except Exception as exc:
            # OOM 或其他异常：通知所有活跃请求，避免客户端永久挂起
            print(f"[Engine Error] {type(exc).__name__}: {exc}")
            with self._queues_lock:
                for q in self._output_queues.values():
                    q.put(_Sentinel(finish_reason="error", error=str(exc)))
            # 释放资源
            for req_id in list(scheduler.running.keys()):
                scheduler.mark_finished(scheduler.running[req_id])
                adapter.free(req_id)
            if _is_mps:
                torch.mps.empty_cache()
            self._running = False

    def _run_engine_loop(self, scheduler, adapter, active_processors, _is_mps):
        """引擎主循环的实际逻辑，用 torch.no_grad() 包裹以避免存储计算图。

        没有 torch.no_grad() 时，PyTorch 会为每次 forward 保存完整的计算图
        （所有中间 tensor 的引用），28 层 Qwen3 的中间 tensor 加起来轻松超过
        40 GB，导致 MPS OOM。cli.py 的 generate 路径已有 no_grad，
        这里是 AsyncEngine 必须补上的对应。
        """
        with torch.no_grad():
            while self._running:
                self._engine_loop_step(scheduler, adapter, active_processors, _is_mps)

    def _engine_loop_step(self, scheduler, adapter, active_processors, _is_mps):
        """引擎主循环的单步执行逻辑。"""
        # ── 1. 接收新请求 ──
        while not self._pending_queue.empty():
            try:
                pending = self._pending_queue.get_nowait()
                scheduler.submit(pending.request_state)
                active_processors[pending.request_state.request_id] = pending.sampling_processor
            except queue.Empty:
                break

        # ── 2. 没有活跃请求就等待 ──
        if not scheduler.has_unfinished():
            import time

            time.sleep(0.001)
            return

        # ── 3. Admit + Allocate（复用 batch_generate_loop 的逻辑）──
        # 从 scheduler.waiting 中取出可以被调度的请求：
        #   - can_admit：检查是否有足够的 KV cache blocks
        #   - can_admit_with_cache：检查 prefix caching 命中情况
        #     - 返回 -1：cache 正在计算中，本轮跳过
        #     - 返回 >0：有 prefix 命中，只需分配剩余的 blocks
        #     - 返回 0：无命中，正常分配
        admitted = []
        while scheduler.waiting:
            req = scheduler.waiting[0]
            prompt_len = req.prompt_ids.shape[1]
            if not adapter.can_admit(prompt_len):
                break
            prompt_ids_list = req.prompt_ids.squeeze(0).tolist()
            num_cached = adapter.can_admit_with_cache(prompt_ids_list)
            if num_cached == -1:
                break
            scheduler.waiting.popleft()
            req.status = RequestStatus.RUNNING
            scheduler.running[req.request_id] = req
            admitted.append(req)
            if num_cached > 0:
                adapter.allocate_with_cache(req.request_id, prompt_ids_list, num_cached)
            else:
                adapter.allocate(req.request_id, prompt_len)

        # ── 4. Batched Prefill ──
        if admitted:
            input_ids, positions = _build_prefill_batch(admitted, self.device)
            batch_request_ids = [req.request_id for req in admitted]
            metadata = adapter.make_prefill_metadata(
                input_ids, positions, request_ids=batch_request_ids
            )
            with set_forward_context(metadata):
                logits = self.model(input_ids, positions=positions)

            # 逐请求采样 + emit 首个 token
            for i, req in enumerate(admitted):
                # M5: hash_blocks 注册
                if hasattr(adapter, "cache") and hasattr(adapter.cache, "hash_blocks"):
                    adapter.cache.hash_blocks(req.request_id, req.prompt_ids.squeeze(0).tolist())
                plen = req.prompt_ids.shape[1]
                # 用该请求的采样器采样首个 token
                proc = active_processors.get(req.request_id)
                req_logits = logits[i, plen - 1, :].unsqueeze(0)  # [1, V]
                if proc is not None:
                    # prefill 阶段不需要 repetition penalty（还没有 generated tokens）
                    proc.set_generated_ids([])
                    token = proc(req_logits)
                else:
                    token = torch.argmax(req_logits, dim=-1, keepdim=True)
                req.last_token = token  # [1, 1]
                req.generated_tokens.append(token)
                req.num_generated = 1
                req.seq_len = plen
                # emit token 到 output queue
                token_id = token.item()
                self._emit_token(req.request_id, token_id)

            adapter.set_seq_lens(admitted)

            # 释放 prefill 阶段的大 tensor，避免 decode 阶段 OOM
            # prefill 的 logits [B, T, V] 在 T 较大时占用很大（如 T=1000 → ~600 MB）
            del input_ids, positions, logits, metadata, batch_request_ids
            # MPS 释放临时显存
            if _is_mps:
                torch.mps.empty_cache()

        # ── 5. Decode（所有 running 请求并行一步）──
        # continuous batching：所有 running 请求（包括刚 prefill 完的）
        # 共享同一次 batched decode forward。每个请求只输入 1 个 token（上一步的输出），
        # 通过 PagedAttention 的 block_table 访问各自的 KV cache。
        running = list(scheduler.running.values())
        if not running:
            return

        adapter.prepare_decode([req.request_id for req in running])
        next_tokens, positions = _build_decode_batch(running, self.device)
        metadata = adapter.make_decode_metadata(next_tokens, positions)

        with set_forward_context(metadata):
            logits = self.model(next_tokens, positions=positions)

        # 采样后立即释放 decode forward 的临时 tensor
        # logits [B, V] 本身很小（~600 KB），但它是 model forward 计算图的根节点，
        # 释放它可以触发 GC 回收整个 forward 过程中所有不再被引用的中间 tensor
        decode_logits = logits[:, -1, :]  # [B, V]

        # 逐请求处理：因为每个请求可能有不同的 SamplingParams
        finished_reqs = []
        for idx, req in enumerate(running):
            req_logit = decode_logits[idx, :].unsqueeze(0)  # [1, V]
            proc = active_processors.get(req.request_id)
            if proc is not None:
                # 注入已生成 token ids（用于 repetition penalty）
                gen_ids = [t.item() for t in req.generated_tokens]
                proc.set_generated_ids([gen_ids])
                token = proc(req_logit)
            else:
                token = torch.argmax(req_logit, dim=-1, keepdim=True)

            tok = token.squeeze(0)  # [1] → scalar
            req.last_token = token  # [1, 1]
            req.generated_tokens.append(token)
            req.num_generated += 1
            req.seq_len += 1

            # M5: hash_blocks 注册
            if hasattr(adapter, "cache") and hasattr(adapter.cache, "hash_blocks"):
                all_ids = req.prompt_ids.squeeze(0).tolist() + [
                    t.item() for t in req.generated_tokens
                ]
                adapter.cache.hash_blocks(req.request_id, all_ids)

            # 完成条件：max_new_tokens 到达 或 EOS
            token_id = tok.item()
            is_done = req.num_generated >= req.max_new_tokens or (
                req.eos_token_id is not None and token_id == req.eos_token_id
            )

            if is_done:
                # finish_reason 区分：EOS 触发 → "stop"，长度到达 → "length"
                finish_reason = (
                    "stop"
                    if (req.eos_token_id is not None and token_id == req.eos_token_id)
                    else "length"
                )
                # EOS token 本身不 emit 给客户端（解码时会被 skip_special_tokens 过滤，
                # 但不 emit 可以避免客户端看到一个空字符串 chunk）
                if not (req.eos_token_id is not None and token_id == req.eos_token_id):
                    self._emit_token(req.request_id, token_id)
                # 发送 _Sentinel 信号，告知客户端生成结束
                self._emit_sentinel(req.request_id, finish_reason)
                finished_reqs.append(req)
            else:
                self._emit_token(req.request_id, token_id)

        # 清理完成的请求
        for req in finished_reqs:
            scheduler.mark_finished(req)
            adapter.free(req.request_id)
            active_processors.pop(req.request_id, None)
        # 有请求完成时清理一次显存（而非每步 decode 都清理）
        if finished_reqs and _is_mps:
            torch.mps.empty_cache()

        # ── 6. 显式释放 decode 阶段的临时 tensor ──
        # 虽然 torch.no_grad() 不存储计算图，但 Python 变量仍持有 tensor 引用
        # 在下一步 forward 前释放，避免新旧 tensor 同时占用显存
        del logits, decode_logits, next_tokens, positions, metadata

    def _emit_token(self, request_id: str, token_id: int) -> None:
        """将 token 推入请求的 output queue（线程安全）。"""
        with self._queues_lock:
            q = self._output_queues.get(request_id)
        if q is not None:
            q.put(token_id)

    def _emit_sentinel(self, request_id: str, finish_reason: str) -> None:
        """将完成信号推入请求的 output queue。"""
        with self._queues_lock:
            q = self._output_queues.get(request_id)
        if q is not None:
            q.put(_Sentinel(finish_reason=finish_reason))

    def cleanup_request(self, request_id: str) -> None:
        """清理请求的 output queue（在 stream 结束后调用）。"""
        with self._queues_lock:
            self._output_queues.pop(request_id, None)
