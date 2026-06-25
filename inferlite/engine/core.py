"""Minimal single-step inference engine and greedy generate loop.

`EngineCore` 是推理流程调度层，不直接实现神经网络计算，也不直接实现采样策略。
它负责把已经完成的各组件串起来：

    model(input_ids, logits_to_keep=1) -> logits [B, 1, V]
    logits[:, -1, :] -> next_token_logits [B, V]
    sampler(next_token_logits) -> next_token [B, 1]

`generate()` 支持 M1（无 cache）和 M2（prefill/decode 两阶段）两条路径：

    # M1：每步 full forward，正确但慢
    generate(engine, input_ids, max_new_tokens=10)

    # M2：prefill 一次处理 prompt，decode 每步只处理 1 个 token
    cache = KVCache.from_config(config, batch_size=1, max_seq_len=512,
                                dtype=torch.float32, device="cpu")
    generate(engine, input_ids, max_new_tokens=10, kv_cache=cache)

注意：
- `logits_to_keep=1` 优化已生效：模型只计算最后一个位置的 lm_head 输出。
- 调用 `generate()` 的上层（如 CLI）应保证在 `torch.no_grad()` 上下文里运行，
  避免构建不必要的梯度计算图。
"""

import torch

from inferlite.engine.protocol import LLMModel
from inferlite.model.kv_cache import KVCache
from inferlite.sampler.greedy import GreedySampler


class EngineCore:
    """最小单步推理引擎。

    EngineCore 只依赖 `LLMModel` 协议，不绑定具体模型类，例如 `Qwen3ForCausalLM`。
    因此只要一个对象能 `model(input_ids) -> logits`，就可以被 EngineCore 使用。
    """

    def __init__(self, model: LLMModel, sampler: GreedySampler) -> None:
        self.model: LLMModel = model
        self.sampler: GreedySampler = sampler

    def step(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行一步 greedy decode。

        Args:
            input_ids: token ids，shape 为 [B, T]。

        Returns:
            next_token: 下一 token ids，shape 为 [B, 1]。
        """
        # logits_to_keep=1：模型只计算最后一个 token 位置的 lm_head 输出，
        # 省去前 T-1 个位置的投影，节约内存和计算量（T12-pre 优化）。
        logits = self.model(input_ids, logits_to_keep=1)
        # logits 形状为 [B, 1, V]，取 [:, -1, :] 得到 [B, V] 交给 sampler。
        next_token_logits = logits[:, -1, :]

        # sampler 只负责 [B, V] -> [B, 1]，不关心 logits 来自哪个模型或哪个位置。
        next_token = self.sampler(next_token_logits)
        return next_token


def generate(
    engine: EngineCore,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    kv_cache: KVCache | None = None,
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
    if kv_cache is None:
        # M1 路径：逻辑不变
        for _ in range(max_new_tokens):
            next_token = engine.step(input_ids)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # EOS 停止：当前 batch 所有序列在当前步生成了 EOS token 时退出。
            # M1 简化：batch=1，只检查当前步是否为 EOS；没有记录每条序列是否曾经输出过 EOS。
            # 多序列 batch 需要 done mask 记录每条序列状态，防止先到 EOS 的序列继续生成无效 token，留 M3。
            # TODO(M3): 支持 done mask 实现真正的每序列 EOS 提前停止。
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids

    # M2 路径：prefill + decode loop
    kv_cache.reset()

    # ----- Prefill -----
    # 一次性把整个 prompt 跑完，所有层的 K/V 写入 cache。
    # position_ids 从 0 开始连续编号，与训练时 full-sequence 前向一致。
    T_p = input_ids.shape[1]
    position_ids = torch.arange(T_p, device=input_ids.device).unsqueeze(0)  # [1, T_p]
    logits = engine.model(input_ids, position_ids=position_ids, kv_cache=kv_cache)
    # 显式更新 cur_len：prefill 写入了 T_p 个 token 的 KV，下一步 decode 从 T_p 位置开始。
    # cur_len 在 generate loop 里维护，不在 model 内部更新（ADR-02：避免模型内部隐式状态）。
    kv_cache.cur_len = T_p

    # 采样 prefill 最后一个位置的 token，作为 decode 第一步的输入。
    next_token = engine.sampler(logits[:, -1, :])
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
        logits = engine.model(next_token, position_ids=pos, kv_cache=kv_cache)
        kv_cache.cur_len += 1
        next_token = engine.sampler(logits[:, -1, :])
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids
