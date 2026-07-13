"""Engine-facing model protocol.

`engine` 层不应该直接绑定某一个具体模型类，比如 `Qwen3ForCausalLM`。
它真正需要的能力很小：

    input_ids [B, T] -> logits [B, T, V]

因此这里用 `Protocol` 定义一个结构化类型：只要某个对象支持
`model(input_ids)` 并返回 logits Tensor，它就可以被 EngineCore 当作 LLMModel 使用。

M2-T4 在原有基础上扩展了 `position_ids` 和 `kv_cache` 两个可选参数：
- `position_ids`：decode 阶段需要传绝对位置，而不是从 0 重新计数。
- `kv_cache`：有 cache 时触发 prefill/decode 两阶段逻辑；None 时走 M1 full forward。

M3-T4 新增 `cache_slots` 和 `cache_positions` 两个可选参数：
- `cache_slots`：每个请求对应的 slot id（batched decode 时按 slot 分别读写 KV）。
- `cache_positions`：每个请求的当前写入位置（per-slot 独立 seq_len）。

注意：
- `LLMModel` 不是模型实现，不会被实例化。
- `__call__` 里的 `...` 不是 TODO，而是"只声明接口，不实现逻辑"。
- 真实逻辑由具体模型提供，例如 `Qwen3ForCausalLM.forward`。
- FakeModel 只要实现 `__call__`，也能在单测里满足这个协议。
"""

from typing import Protocol

import torch


class LLMModel(Protocol):
    """最小 LLM 推理协议：input_ids -> logits，支持可选的 logits_to_keep / position_ids / kv_cache。

    这个协议描述的是 EngineCore 对模型的最低要求：

        # M1 full forward（无 cache）
        logits = model(input_ids)
        logits = model(input_ids, logits_to_keep=1)

        # M2 prefill
        logits = model(input_ids, position_ids=pos, kv_cache=cache)

        # M2 decode（每步只传 1 个 token）
        logits = model(next_token, position_ids=abs_pos, kv_cache=cache)

    为什么定义 `__call__` 而不是 `forward`？
    - EngineCore 实际会调用 `model(input_ids)`。
    - PyTorch 的 `nn.Module.__call__` 会转发到具体模型的 `forward`。
    - 普通 FakeModel 也可以直接实现 `__call__`，不必继承 `nn.Module`。
    """

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        logits_to_keep: int | None = None,
        position_ids: torch.Tensor | None = None,
        kv_cache: object = None,
        cache_slots: torch.Tensor | None = None,
        cache_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回 logits。

        Args:
            input_ids: [B, T] 形状的 token ids。
            logits_to_keep: 若为非 None，只返回最后 logits_to_keep 个位置的 logits。
            position_ids: [B, T]，绝对位置。None 时模型内部自动生成 0..T-1。
                decode 阶段必须传绝对位置（如 [[cur_len]]），否则 RoPE 每步都在位置 0。
            kv_cache: 全模型 KVCache。None 走 M1 full attention；非 None 走 M2/M3 两阶段。
            cache_slots: M3 batched decode 专用。[B] 每个请求对应的 slot id。
            cache_positions: M3 batched decode 专用。[B] 每个请求的当前写入位置。

        Returns:
            logits: [B, T, V] 或 [B, logits_to_keep, V]。
        """
        ...
