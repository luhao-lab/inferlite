"""M3 Continuous Batching generate。

`batch_generate()` 是 M3 的核心入口，串起 scheduler + BatchedKVCache + batched attention，
实现 continuous batching 的最小执行流。

与 M2 `generate()` 的关系：
  - M2 `generate()`：单请求，prefill + decode 两阶段
  - M3 `batch_generate()`：多请求，continuous batching

调用示例：
    outputs = batch_generate(
        model, sampler,
        prompts=[ids_a, ids_b, ids_c],
        max_new_tokens=16,
        max_num_slots=2,
        config=config,
        max_seq_len=512,
    )
"""

import torch

from inferlite.config import ModelConfig
from inferlite.engine.protocol import LLMModel
from inferlite.model import BatchedKVCache
from inferlite.sampler.greedy import GreedySampler
from inferlite.scheduler.fcfs import FCFSScheduler
from inferlite.scheduler.request import RequestState


def batch_generate(
    model: LLMModel,
    sampler: GreedySampler,
    prompts: list[torch.Tensor],
    max_new_tokens: int,
    max_num_slots: int,
    config: ModelConfig,
    max_seq_len: int,
    eos_token_id: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """M3 continuous batching generate。

    Args:
        model: 推理模型（Qwen3ForCausalLM 或符合 LLMModel 协议的对象）。
        sampler: 采样器。
        prompts: 多个 prompt，每个 shape 为 [1, T_i]。
        max_new_tokens: 每个请求最多生成的新 token 数。
        max_num_slots: KV cache 的最大槽位数（= 最大并发请求数）。
        config: 模型配置，用于创建 BatchedKVCache。
        max_seq_len: 每个请求的最大序列长度。
        eos_token_id: EOS token id，生成到时提前停止。
        device: 计算设备。
        dtype: 数据类型。

    Returns:
        每个请求的生成结果列表（按 request_id 排序），
        每个元素为 prompt + generated token ids，shape [1, T_i + n_i]。
    """
    # ── 初始化 scheduler：所有请求先进 waiting 队列 ──
    # waiting 请求不占 KV slot，只有 admit 到 running 后才分配 slot。
    scheduler = FCFSScheduler(max_num_seqs=max_num_slots)
    for i, prompt_ids in enumerate(prompts):
        req = RequestState(
            request_id=str(i),
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        scheduler.submit(req)

    # ── 创建 BatchedKVCache：固定 S 个 slot，每个 slot 存 max_seq_len 个 token ──
    cache = BatchedKVCache.from_config(
        config=config,
        max_num_slots=max_num_slots,
        max_seq_len=max_seq_len,
        dtype=dtype,
        device=device,
    )

    # ── 主循环：iteration-level scheduling ──
    # 每轮迭代：admit 新请求 → prefill → batched decode → 更新状态
    # finished 请求在 step 3 释放 slot，下一轮 admit 时新请求自动进入。
    while scheduler.has_unfinished():
        # ── 1. admit + prefill ──
        # admit_until_full() 只返回本轮新 admit 的请求（之前已在 running 的不会重复返回）。
        # 逐条 prefill：每个请求独立跑一次 full forward，KV 写入对应的 slot。
        admitted = scheduler.admit_until_full()
        for request in admitted:
            slot = cache.allocate_slot(request.request_id)
            request.slot_id = slot

            prompt_len = request.prompt_ids.shape[1]
            position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)  # [1, T_p]
            # prefill：整条 prompt 一次前向，KV 写入 cache slot
            # cache_slots=[slot]：告诉 attention 层写入哪个 slot（B=1）
            logits = model(
                request.prompt_ids,
                position_ids=position_ids,
                kv_cache=cache,
                cache_slots=torch.tensor([slot]),
            )

            # prefill 后采样第一个 token（作为 decode 第一步的输入）
            request.seq_len = prompt_len
            request.last_token = sampler(logits[:, -1, :])  # [1, 1]
            request.generated_tokens.append(request.last_token)
            request.num_generated = 1
            cache.seq_lens[slot] = prompt_len

        # ── 2. batched decode one step ──
        # 把所有 running 请求组成一个 batch，并行执行一步 decode。
        # 每个请求的 cache_position 独立（= 该请求的 seq_len），
        # attention 层通过 cache_slots + cache_positions 分别读写各自的 KV。
        if not scheduler.running:
            break
        running = list(scheduler.running.values())
        cache_slots = torch.tensor([req.slot_id for req in running])
        # cache_positions，每个请求当前的写入位置（= seq_len）
        cache_positions = cache.seq_lens[cache_slots]  # [B]
        position_ids = cache_positions.unsqueeze(1)  # [B, 1]
        # next_tokens: [B, 1]，拼接每个请求上一步的 last_token
        next_tokens = torch.cat(
            [req.last_token for req in running if req.last_token is not None], dim=0
        )
        logits = model(
            next_tokens,
            position_ids=position_ids,
            kv_cache=cache,
            cache_slots=cache_slots,
            cache_positions=cache_positions,
        )

        # ── 3. sample + update state + finish ──
        sampled = sampler(logits[:, -1, :])
        for request, next_token in zip(running, sampled, strict=False):
            # next_token: [1]（1D），unsqueeze 为 [1, 1] 保持与 prefill 阶段一致
            request.last_token = next_token.unsqueeze(0)
            request.generated_tokens.append(next_token.unsqueeze(0))
            request.num_generated += 1
            request.seq_len += 1
            cache.seq_lens[request.slot_id] = request.seq_len

            # 完成条件：max_new_tokens 到达 或 EOS
            is_max = request.num_generated >= request.max_new_tokens
            is_eos = eos_token_id is not None and next_token.item() == eos_token_id
            if is_max or is_eos:
                scheduler.mark_finished(request)
                # 释放 slot：下一轮循环 admit_until_full 就能看到空闲 slot
                cache.free_slot(request.request_id)

    # ── 收集结果（按 request_id 排序，保证与输入 prompts 顺序一致）──
    results = []
    for req_id in sorted(scheduler.finished.keys(), key=int):
        req = scheduler.finished[req_id]
        results.append(torch.cat([req.prompt_ids] + req.generated_tokens, dim=1))
    return results
