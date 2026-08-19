"""公共 batch generation 主循环（对齐 vLLM V1 execute_model）。

统一 M3/M4 的 batch generation 逻辑，通过 CacheAdapter 屏蔽 cache 差异。

执行流程（每轮迭代）：
  1. Admit:     adapter.can_admit() 检查容量 → scheduler 取请求 → adapter.allocate() 分配 cache
  2. Prefill:   _build_prefill_batch() 拼 batch → adapter.make_prefill_metadata() 构造元数据
                → set_forward_context(metadata) → model forward → 采样首个 token
  3. Decode:    adapter.prepare_decode() 分配新 token 空间
                → _build_decode_batch() 拼 batch → adapter.make_decode_metadata() 构造元数据
                → set_forward_context(metadata) → model forward → 采样 → 更新状态
                → 完成则 adapter.free()
  4. 收集结果:  按 request_id 排序返回

与 vLLM V1 的对应：
  vLLM model_runner.execute_model()  → batch_generate_loop()
  vLLM scheduler.schedule()          → _admit()
  vLLM input_batch                   → _build_prefill_batch / _build_decode_batch
  vLLM set_forward_context()         → set_forward_context()（同名）

T7-A10 后：M3/M4 均使用 batched prefill（padded → 一次 forward）。
"""

import time

import torch

from inferlite.engine.context import set_forward_context
from inferlite.scheduler.request import RequestStatus


def _admit(scheduler, adapter):
    """从 waiting 队列取请求到 running，直到 adapter 容量不够。

    FCFS 策略：队首不够空间就停，不跳过头部请求（对齐 paged_core._paged_admit）。
    与 M3 scheduler.admit_until_full() 的区别：M3 只看 max_num_seqs（slot 数），
    这里通过 adapter.can_admit() 统一处理 M3 的 slot 检查和 M4 的 block 检查。

    Returns:
        本轮新 admit 的请求列表（之前已在 running 的不会重复返回）。
    """
    admitted = []
    while scheduler.waiting:
        req = scheduler.waiting[0]
        # 容量检查：M3 看 slot 空闲数，M4 看 block 是否够放 prompt
        if not adapter.can_admit(req.prompt_ids.shape[1]):
            break
        # 从 waiting 移到 running
        scheduler.waiting.popleft()
        req.status = RequestStatus.RUNNING
        scheduler.running[req.request_id] = req
        admitted.append(req)
    return admitted


def _build_prefill_batch(admitted, device):
    """拼接 admitted 请求为 padded batched prefill 输入。

    多个请求的 prompt 长度不同，pad 到最长。padding 位置为 0，
    但 attention 的 valid_lens_mask 会忽略 padding 部分的 KV 写入。

    Args:
        admitted: 本轮新 admit 的请求列表
        device: 计算设备

    Returns:
        (input_ids [B, max_len], positions [B, max_len])
        positions 每行 [0, 1, ..., plen-1, 0, 0, ...]
    """
    prompt_lens = [req.prompt_ids.shape[1] for req in admitted]
    max_len = max(prompt_lens)
    B = len(admitted)
    input_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    positions = torch.zeros(B, max_len, dtype=torch.long, device=device)
    for i, req in enumerate(admitted):
        plen = prompt_lens[i]
        input_ids[i, :plen] = req.prompt_ids.squeeze(0)  # [1, T] → [T]
        positions[i, :plen] = torch.arange(plen, device=device)
    return input_ids, positions


def _build_decode_batch(running, device):
    """拼接 running 请求的 last_token 为 decode batch。

    每个请求只输入上一步采样的 1 个 token（不是全量历史），
    历史 KV 从 cache 读取。

    Args:
        running: 当前 running 的请求列表
        device: 计算设备

    Returns:
        (next_tokens [B, 1], positions [B, 1])
        positions = req.seq_len（当前写入位置，prepare_decode 之后、sample 之前）
    """
    # 拼接每个请求上一步采样的 token 作为本步输入
    next_tokens = torch.cat(
        [req.last_token for req in running if req.last_token is not None], dim=0
    )
    # 每个请求的当前写入位置 = 已写入 token 数（seq_len）
    positions = torch.tensor([req.seq_len for req in running], dtype=torch.long, device=device)
    return next_tokens, positions.unsqueeze(1)  # [B, 1]


def batch_generate_loop(
    model,
    sampler,
    scheduler,
    adapter,
    prompts,
    max_new_tokens,
    eos_token_id=None,
    device="cpu",
    metrics=None,
):
    """公共 batch generation 主循环。

    初始化时调用 adapter.bind_kv_cache(model) 绑定 cache 到 Attention 层，
    然后进入 iteration-level scheduling 主循环。

    Args:
        model: 推理模型（Qwen3ForCausalLM）
        sampler: 采样器（GreedySampler）
        scheduler: FCFS 调度器（已 submit 所有请求到 waiting）
        adapter: CacheAdapter 实例（NoCache/Single/Batched/Paged）
        prompts: 未使用（请求已在 scheduler 中），保留参数兼容
        max_new_tokens: 每个请求最多生成的新 token 数
        eos_token_id: EOS token id，生成到时提前停止
        device: 计算设备
        metrics: 可选的 MetricsCollector

    Returns:
        每个请求的生成结果列表（按 request_id 排序），
        每个元素为 prompt + generated token ids，shape [1, T_i + n_i]。
    """
    # ── 初始化：绑定 cache 到模型的每层 Attention ──
    # 对齐 vLLM V1 worker 初始化时的 kv_cache 赋值
    adapter.bind_kv_cache(model)

    while scheduler.has_unfinished():
        # ── 1. Admit + Allocate（逐条检查容量，避免超额分配）──
        admitted = []
        while scheduler.waiting:
            req = scheduler.waiting[0]
            prompt_len = req.prompt_ids.shape[1]
            if not adapter.can_admit(prompt_len):  # ← 先检查容量（M2/M3/M4 通用）
                break
            # M5: 先查 prefix cache
            prompt_ids = req.prompt_ids.squeeze(0).tolist()
            num_cached = adapter.can_admit_with_cache(prompt_ids)
            if num_cached == -1:
                break  # 容量不够
            scheduler.waiting.popleft()
            req.status = RequestStatus.RUNNING
            scheduler.running[req.request_id] = req
            admitted.append(req)
            if num_cached > 0:
                adapter.allocate_with_cache(req.request_id, prompt_ids, num_cached)
            else:
                adapter.allocate(req.request_id, prompt_len)  # M4 路径
            if metrics:
                metrics.record_arrival(req.request_id)
                metrics.record_prompt_tokens(req.request_id, prompt_len)
                metrics.record_scheduled(req.request_id)

        # ── 2. Batched Prefill（所有新请求 padded 成 batch，一次 forward）──
        if admitted:
            input_ids, positions = _build_prefill_batch(admitted, device)
            batch_request_ids = [req.request_id for req in admitted]
            metadata = adapter.make_prefill_metadata(
                input_ids, positions, request_ids=batch_request_ids
            )
            with set_forward_context(metadata):
                logits = model(input_ids, positions=positions)
            # 逐请求采样：取每个请求最后一个真实 token 的 logits
            for i, req in enumerate(admitted):
                if hasattr(adapter, "cache") and hasattr(adapter.cache, "hash_blocks"):
                    adapter.cache.hash_blocks(req.request_id, req.prompt_ids.squeeze(0).tolist())
                plen = req.prompt_ids.shape[1]
                req.last_token = sampler(logits[i, plen - 1, :].unsqueeze(0))  # [1, 1]
                req.generated_tokens.append(req.last_token)
                req.num_generated = 1
                req.seq_len = plen
            adapter.set_seq_lens(admitted)
            if metrics:
                for req in admitted:
                    metrics.record_first_token(req.request_id)

        # ── 3. Decode ──
        # 所有 running 请求（包括刚 prefill 完的）并行执行一步 decode
        running = list(scheduler.running.values())
        if not running:
            break

        # 3a. 分配新 token 的 cache 空间（仅 PagedCacheAdapter 实际做事：append_token）
        adapter.prepare_decode([req.request_id for req in running])

        # 3b. 构造 decode batch 和 metadata
        next_tokens, positions = _build_decode_batch(running, device)
        metadata = adapter.make_decode_metadata(next_tokens, positions)

        # 3c. model forward（metadata 通过 ForwardContext 传给 Attention 层）
        decode_start = time.perf_counter()
        with set_forward_context(metadata):
            logits = model(next_tokens, positions=positions)
        decode_ms = (time.perf_counter() - decode_start) * 1000

        # 3d. 采样 + 更新状态 + 完成检查
        sampled = sampler(logits[:, -1, :])
        for req, tok in zip(running, sampled, strict=False):
            req.last_token = tok.unsqueeze(0)  # [1] → [1, 1]
            req.generated_tokens.append(tok.unsqueeze(0))
            req.num_generated += 1
            req.seq_len += 1
            if hasattr(adapter, "cache") and hasattr(adapter.cache, "hash_blocks"):
                all_ids = req.prompt_ids.squeeze(0).tolist() + [
                    t.item() for t in req.generated_tokens
                ]
                adapter.cache.hash_blocks(req.request_id, all_ids)
            # 完成条件：max_new_tokens 到达 或 EOS
            is_done = req.num_generated >= req.max_new_tokens or (
                eos_token_id is not None and tok.item() == eos_token_id
            )
            if is_done:
                scheduler.mark_finished(req)
                adapter.free(req.request_id)  # 释放 cache 空间
                if metrics:
                    metrics.record_output_tokens(req.request_id, req.num_generated)
                    metrics.record_finished(req.request_id)

        if metrics:
            step_idx = len(metrics.step_metrics)
            metrics.record_step(
                step_idx=step_idx,
                batch_size=len(running),
                max_seq_len=int(positions.max().item()),
                decode_ms=decode_ms,
                output_tokens=len(running),
                running_count=len(scheduler.running),
                waiting_count=len(scheduler.waiting),
                occupied_slots=len(running),
            )

    # ── 收集结果（按 request_id 排序，保证与输入 prompts 顺序一致）──
    return [
        torch.cat([req.prompt_ids] + req.generated_tokens, dim=1)
        for req_id in sorted(scheduler.finished, key=int)
        for req in [scheduler.finished[req_id]]
    ]
