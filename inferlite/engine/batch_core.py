import torch

from inferlite.engine.protocol import LLMModel
from inferlite.model import BatchedKVCache
from inferlite.sampler.greedy import GreedySampler


def batch_generate(
    model: LLMModel,
    sampler: GreedySampler,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    kv_cache: BatchedKVCache | None = None,
) -> torch.Tensor:
    """用 `EngineCore.step` 做最小 greedy generate loop，支持 EOS 提前停止。

    Args:
        engine: 已经组装好 model + sampler 的单步推理引擎。
        input_ids: prompt token ids，shape 为 [B, T]。
        max_new_tokens: 最多生成多少个新 token（硬上限）。
        eos_token_id: EOS token 的 id。当生成的 token 等于 eos_token_id 时提前停止。
            设为 None 时不做 EOS 检查，严格跑满 max_new_tokens 步（向后兼容）。
        kv_cache: 全模型 KVCache。
            None：走 M1 full forward，每步重跑完整序列（向后兼容）。
            非 None：走 M2 两阶段——
                Prefill：一次性处理整个 prompt，KV 写入 cache。
                Decode：每步只传 1 个 token，历史 KV 从 cache 读取。

    Returns:
        output_ids: prompt + generated token ids，shape 为 [B, T + n]，
            其中 n <= max_new_tokens。若提前遇到 EOS，n 可能小于 max_new_tokens。

    调用方应在 `torch.no_grad()` 上下文里使用此函数，避免构建不必要的梯度图。
    这个函数只负责 token id 级别的循环，不负责 tokenizer encode/decode。
    CLI 会在外层完成文本与 token id 的转换。
    """

    # 这里可能会有非None的情况吗？
    if kv_cache is not None:
        kv_cache.reset_slots()

    # ----- Prefill -----
    # 一次性把整个 prompt 跑完，所有层的 K/V 写入 cache。
    # position_ids 从 0 开始连续编号，与训练时 full-sequence 前向一致。
    T_p = input_ids.shape[1]
    position_ids = torch.arange(T_p, device=input_ids.device).unsqueeze(0)  # [1, T_p]
    logits = model(input_ids, position_ids=position_ids, kv_cache=kv_cache)
    # 显式更新 cur_len：prefill 写入了 T_p 个 token 的 KV，下一步 decode 从 T_p 位置开始。
    # cur_len 在 generate loop 里维护，不在 model 内部更新（ADR-02：避免模型内部隐式状态）。
    kv_cache.cur_len = T_p

    # 采样 prefill 最后一个位置的 token，作为 decode 第一步的输入。
    next_token = sampler(logits[:, -1, :])
    input_ids = torch.cat([input_ids, next_token], dim=1)

    # ----- Decode Loop -----
    for _ in range(max_new_tokens - 1):
        # EOS 检查放在 loop 开头：检查上一步（prefill 采样或上一 decode 步）生成的 token。
        # 这样既不漏掉 EOS，也不会在 EOS 之后多生成一步。
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
        # position_ids 必须是绝对位置（cur_len），不能从 0 重新计数。
        # 用 [[0]] 是沉默 bug：RoPE 认为每步都在位置 0，输出质量下降但不报错（ADR-04）。
        pos = torch.tensor([[kv_cache.cur_len]], device=input_ids.device)  # [1, 1] 绝对位置
        logits = model(next_token, position_ids=pos, kv_cache=kv_cache)
        kv_cache.cur_len += 1
        next_token = sampler(logits[:, -1, :])
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
