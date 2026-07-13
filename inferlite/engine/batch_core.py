import torch

from inferlite.engine.protocol import LLMModel
from inferlite.sampler.greedy import GreedySampler


class BatchEngineCore:
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
