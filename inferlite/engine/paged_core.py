"""M4-T5 Paged BatchEngine：用 PagedKVCache 替换 fixed-slot cache 的 batch generation。

与 M3 batch_core.py 的对应关系：
- batch_generate     → batch_generate_paged
- max_num_slots      → num_blocks × block_size
- BatchedKVCache     → PagedKVCache
- allocate_slot      → allocate_request
- free_slot          → free_request
- cache_slots        → request_ids
- cache_positions    → block_tables[rid].seq_len
- kv_cache           → paged_kv_cache

主循环结构不变：admit → prefill → decode → finish/free。
区别仅在 KV 管理层从 fixed-slot 换成 paged block。
"""

import time

import torch

from inferlite.cache import PagedKVCache
from inferlite.config import ModelConfig
from inferlite.engine.metrics import MetricsCollector
from inferlite.engine.protocol import LLMModel
from inferlite.sampler.greedy import GreedySampler
from inferlite.scheduler.fcfs import FCFSScheduler
from inferlite.scheduler.request import RequestState, RequestStatus


def batch_generate_paged(
    model: LLMModel,
    sampler: GreedySampler,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    num_blocks: int,  # ← paged: 物理 block 总数
    block_size: int,  # ← paged: 每个 block 的 token 容量
    config: ModelConfig,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    metrics: MetricsCollector | None = None,
) -> list[torch.Tensor]:
    """M4 paged continuous batching generate。

    与 M3 batch_generate 的区别：
    - KV cache 从 fixed-slot BatchedKVCache 换成 paged PagedKVCache
    - admission 从 max_num_seqs 换成 can_allocate(prompt_len)
    - model 参数从 kv_cache/cache_slots/cache_positions 换成 paged_kv_cache/request_ids/is_prefill

    Args:
        model: 推理模型（Qwen3ForCausalLM 或符合 LLMModel 协议的对象）。
        sampler: 采样器。
        prompts: 多个 prompt，每个 shape 为 [1, T_i]。
        max_new_tokens: 每个请求最多生成的新 token 数。
        num_blocks: 物理 block 总数（paged cache 的总容量 = num_blocks × block_size tokens）。
        block_size: 每个 block 的 token 容量（内部分碎片上界 = block_size - 1）。
        config: 模型配置，用于创建 PagedKVCache。
        eos_token_id: EOS token id，生成到时提前停止。
        device: 计算设备。
        dtype: 数据类型。
        metrics: 可选的 MetricsCollector，传入则在执行过程中记录请求级和步级指标。

    Returns:
        每个请求的生成结果列表（按 request_id 排序），
        每个元素为 prompt + generated token ids，shape [1, T_i + n_i]。
    """
    # ── 初始化 scheduler：所有请求先进 waiting 队列 ──
    # max_num_seqs 设为 num_blocks 作为绝对并发上界（每个请求至少占 1 个 block），
    # 实际并发由 _paged_admit 的 can_allocate 控制。
    scheduler = FCFSScheduler(max_num_seqs=num_blocks)
    for i, prompt_ids in enumerate(prompts):
        req = RequestState(
            request_id=str(i),
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        scheduler.submit(req)
        if metrics:
            metrics.record_arrival(req.request_id)
            metrics.record_prompt_tokens(req.request_id, prompt_ids.shape[1])

    # ── 初始化 PagedKVCache：按 block 粒度管理 K/V 存储 ──
    # 与 M3 BatchedKVCache 的区别：
    # - M3: [max_num_slots, n_kv_heads, max_seq_len, head_dim] 固定分配
    # - M4: [num_blocks, block_size, n_kv_heads, head_dim] 按需分配
    # 总容量 = num_blocks × block_size tokens，但分配粒度是 block 而不是整个 slot。
    paged_cache = PagedKVCache.from_config(
        config=config,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )

    # ── 主循环：iteration-level scheduling ──
    # 每轮迭代：admit 新请求 → batched prefill → batched decode → 更新状态
    # finished 请求在 step 3 释放 block，下一轮 admit 时新请求自动进入。
    while scheduler.has_unfinished():
        # ── 1. admit + batched prefill ──
        # _paged_admit 替代 M3 的 admit_until_full()：
        # - M3 只看 max_num_seqs（slot 数量）
        # - M4 额外检查 can_allocate(prompt_len)（block 数量）
        admitted = _paged_admit(scheduler, paged_cache)
        if admitted:
            # 为每个 admitted 请求分配 paged block
            # allocate_request 内部调用 BlockTable.allocate → 从 BlockPool 拿物理 block
            # 只分配 prompt 所需的 block 数，decode 阶段按需追加
            for request in admitted:
                prompt_len = request.prompt_ids.shape[1]
                paged_cache.allocate_request(request.request_id, prompt_len)
                if metrics:
                    metrics.record_scheduled(request.request_id)
                    metrics.record_prefill_start(request.request_id)

            # 拼 batched prefill：多个 admitted 请求 pad 到最长 prompt 长度，一次前向完成
            # T4 attention 层的 _build_valid_lens_mask 已支持变长 mask，
            # padding 部分不会写入 KV cache。
            prompt_lens = [req.prompt_ids.shape[1] for req in admitted]
            max_prompt_len = max(prompt_lens)
            batch_input_ids = torch.zeros(
                len(admitted), max_prompt_len, dtype=torch.long, device=device
            )
            for i, req in enumerate(admitted):
                batch_input_ids[i, : req.prompt_ids.shape[1]] = req.prompt_ids.squeeze(0)

            # position_ids：每行 [0, 1, ..., prompt_len-1, 0, 0, ...]
            # padding 部分位置为 0，但不影响结果（valid_lens_mask 会忽略 padding）
            batch_position_ids = torch.zeros(
                len(admitted), max_prompt_len, dtype=torch.long, device=device
            )
            for i, plen in enumerate(prompt_lens):
                batch_position_ids[i, :plen] = torch.arange(plen, device=device)

            batch_request_ids = [req.request_id for req in admitted]

            # batched prefill 前向：一次 model forward 处理所有 admitted 请求
            # 与 M3 逐条 prefill 不同：M4 合并为一次前向，减少 kernel launch 开销
            logits = model(
                input_ids=batch_input_ids,
                position_ids=batch_position_ids,
                paged_kv_cache=paged_cache,
                request_ids=batch_request_ids,
                is_prefill=True,
            )

            # 逐请求采样第一个 decode token
            # 注意：变长 prefill 时 logits[i, plen-1] 是第 i 个请求最后一个真实 token 的 logits
            # （不是 logits[i, -1]，因为 -1 位置可能是 padding）
            for i, request in enumerate(admitted):
                plen = prompt_lens[i]
                last_logits = logits[i, plen - 1, :].unsqueeze(0)  # [1, V]
                request.seq_len = plen
                request.last_token = sampler(last_logits)  # [1, 1]
                request.generated_tokens.append(request.last_token)
                request.num_generated = 1
                if metrics:
                    metrics.record_prefill_end(request.request_id)
                    metrics.record_first_token(request.request_id)

        # ── 2. batched decode one step ──
        # 不管有没有新 admitted 请求，decode 都要执行（running 里的请求每步都要 decode）
        if not scheduler.running:
            break
        running = list(scheduler.running.values())
        request_ids = [req.request_id for req in running]

        # decode 前先 append_token：让 block_table 的 seq_len +1
        # 如果当前 block 已满（seq_len 跨 block 边界），append_token 会自动分配新 block
        # 注意：block 不足时会抛 RuntimeError（M4 不做抢占，高并发场景留后续里程碑处理）
        for rid in request_ids:
            paged_cache.append_token(rid)

        # position_ids：每个请求当前写入位置（append 后的 seq_len）
        # 与 M3 的区别：
        # - M3: cache_positions = cache.seq_lens[slots]（从 fixed-slot 数组取）
        # - M4: block_tables[rid].seq_len（从 block table 取）
        positions = torch.tensor(
            [paged_cache.block_tables[rid].seq_len - 1 for rid in request_ids],
            dtype=torch.long,
            device=device,
        )
        position_ids = positions.unsqueeze(1)  # [B, 1]

        # next_tokens: [B, 1]，拼接每个请求上一步的 last_token
        next_tokens = torch.cat(
            [req.last_token for req in running if req.last_token is not None], dim=0
        )
        # decode model 前向：与 M3 的区别是参数
        # - M3: kv_cache + cache_slots + cache_positions
        # - M4: paged_kv_cache + request_ids + is_prefill=False
        # layer_idx 由 Qwen3Model 的 enumerate 自动生成，不需要这里传
        decode_start = time.perf_counter()
        logits = model(
            input_ids=next_tokens,
            position_ids=position_ids,
            paged_kv_cache=paged_cache,
            request_ids=request_ids,
            is_prefill=False,
        )
        decode_ms = (time.perf_counter() - decode_start) * 1000

        # ── 3. sample + update state + finish ──
        # 采样后逐请求更新状态（每个 RequestState 是独立对象，必须循环）
        sampled = sampler(logits[:, -1, :])
        for request, next_token in zip(running, sampled, strict=False):
            request.last_token = next_token.unsqueeze(0)
            request.generated_tokens.append(next_token.unsqueeze(0))
            request.num_generated += 1
            request.seq_len += 1

            # 完成条件：max_new_tokens 到达或 EOS
            is_max = request.num_generated >= request.max_new_tokens
            is_eos = eos_token_id is not None and next_token.item() == eos_token_id
            if is_max or is_eos:
                scheduler.mark_finished(request)
                # 释放 paged block：与 M3 free_slot 对应
                # free_request 会释放该请求的所有 block 回 BlockPool
                paged_cache.free_request(request.request_id)
                if metrics:
                    metrics.record_output_tokens(request.request_id, request.num_generated)
                    metrics.record_finished(request.request_id)

        # 记录本 step 指标
        if metrics:
            step_idx = len(metrics.step_metrics)
            max_seq_len_step = int(positions.max().item())
            metrics.record_step(
                step_idx=step_idx,
                batch_size=len(running),
                max_seq_len=max_seq_len_step,
                decode_ms=decode_ms,
                output_tokens=len(running),
                running_count=len(scheduler.running),
                waiting_count=len(scheduler.waiting),
                occupied_slots=len(scheduler.running),
            )

    # ── 收集结果（按 request_id 排序，保证与输入 prompts 顺序一致）──
    results = []
    for req_id in sorted(scheduler.finished.keys(), key=int):
        req = scheduler.finished[req_id]
        results.append(torch.cat([req.prompt_ids] + req.generated_tokens, dim=1))
    return results


def _paged_admit(scheduler: FCFSScheduler, paged_cache: PagedKVCache) -> list[RequestState]:
    """Block-aware admission：只在 cache 有足够 block 时从 waiting 取请求到 running。

    与 M3 FCFSScheduler.admit_until_full() 的区别：
    - M3 只看 max_num_seqs（slot 数量）
    - M4 额外检查 can_allocate(prompt_len)（block 数量）
    """
    admitted = []
    while scheduler.waiting:
        req: RequestState = scheduler.waiting[0]
        prompt_len = req.prompt_ids.shape[1]
        if not paged_cache.can_allocate(prompt_len):
            break  # FCFS：队首不够空间就停，不跳过头部请求
        scheduler.waiting.popleft()
        req.status = RequestStatus.RUNNING
        scheduler.running[req.request_id] = req
        admitted.append(req)
    return admitted
