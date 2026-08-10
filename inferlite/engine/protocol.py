"""Engine-facing model protocol。

`engine` 层不应该直接绑定某一个具体模型类，比如 `Qwen3ForCausalLM`。
它真正需要的能力很小：

    input_ids [B, T] -> logits [B, T, V]

因此这里用 `Protocol` 定义一个结构化类型：只要某个对象支持
`model(input_ids)` 并返回 logits Tensor，它就可以被 EngineCore 当作 LLMModel 使用。

T7-A4 瘦身：
- 从 9 参数精简到 2 参数（input_ids + positions）
- cache 信息通过 adapter.bind_kv_cache() 初始化时绑定，不经 __call__ 传递
- metadata 通过 ForwardContext 全局上下文传递，不经 __call__ 传递
- logits_to_keep 保留，用于 decode 步优化

注意：
- `LLMModel` 不是模型实现，不会被实例化。
- `__call__` 里的 `...` 不是 TODO，而是"只声明接口，不实现逻辑"。
- 真实逻辑由具体模型提供，例如 `Qwen3ForCausalLM.forward`。
- FakeModel 只要实现 `__call__`，也能在单测里满足这个协议。
"""

from typing import Protocol

import torch


class LLMModel(Protocol):
    """最小 LLM 推理协议：input_ids -> logits。

    对齐 vLLM V1 的 model runner 接口（简化版）。

    cache 和 metadata 不经过 __call__ 传递：
    - cache 通过 adapter.bind_kv_cache(model) 在初始化时绑定到 Attention 层
    - metadata 通过 set_forward_context(metadata) 在每次 forward 前设置

    调用示例：

        # M1: 无 cache，每步 full forward
        logits = model(input_ids)

        # M2/M3/M4: cache 已绑定，metadata 已设置
        with set_forward_context(metadata):
            logits = model(input_ids, positions=positions)

        # 只取最后 N 个位置的 logits（decode 优化）
        logits = model(input_ids, logits_to_keep=1)
    """

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        """返回 logits。

        Args:
            input_ids: [B, T] 形状的 token ids。
            positions: [B, T]，绝对位置。None 时模型内部自动生成 0..T-1。
                decode 阶段必须传绝对位置（如 [[cur_len]]），否则 RoPE 每步都在位置 0。
            logits_to_keep: 若为非 None，只返回最后 logits_to_keep 个位置的 logits。

        Returns:
            logits: [B, T, V] 或 [B, logits_to_keep, V]。
        """
        ...

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """从 hidden states 计算 logits（lm_head 投影）。

        对齐 vLLM V1 的 logits 与 forward 分离模式。
        当 forward 返回 hidden_states 时使用；当前 forward 直接返回 logits，
        此方法供未来 A5 阶段使用。
        """
        ...
