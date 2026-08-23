"""扩展采样策略：temperature + top-k + top-p + repetition penalty。

M6 新增，替代 GreedySampler 的单一 argmax 策略。
接口兼容 GreedySampler：__call__(logits) -> [B, 1]，可直接作为 batch_generate_loop 的 sampler 参数。

采样流水线（logits → token）：
  1. repetition penalty：对已生成 token 的 logit 做除法惩罚（Ctrl 2019）
  2. temperature：除以温度参数，控制分布锐度
  3. top-k：只保留概率最高的 k 个 token
  4. top-p (nucleus)：保留累积概率 ≤ p 的最小 token 集合
  5. softmax → multinomial 采样

特殊路径：
  - temperature ≤ 0：退化为 greedy argmax（与 GreedySampler 等价）
  - seed 非空：torch.Generator 可复现采样结果
"""

import torch


class SamplingParams:
    """采样参数配置。

    所有参数都有安全默认值：temperature=0 等价于 greedy，
    不设置 top_k/top_p 则不做过滤，repetition_penalty=1.0 则不惩罚。
    """

    def __init__(
        self,
        temperature: float = 0.0,
        top_k: int = -1,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed


class SamplingProcessor:
    """可配置的 token 采样器，接口兼容 GreedySampler。

    __call__(logits) -> [B, 1]，可以直接替换 GreedySampler 传入 batch_generate_loop。

    使用方式：
        processor = SamplingProcessor(SamplingParams(temperature=0.8, top_p=0.9))
        next_tokens = processor(logits)  # [B, 1]

    repetition penalty 需要知道每个请求已生成的 token ids，
    通过 set_generated_ids() 在每次 decode 前注入：
        processor.set_generated_ids([[1, 5, 23], [7, 12]])  # per-request token lists
        next_tokens = processor(logits)
    """

    def __init__(self, params: SamplingParams | None = None) -> None:
        self.params = params or SamplingParams()
        # torch.Generator 用于可复现的随机采样；seed=None 时使用默认 RNG
        # 初始在 CPU 上创建；__call__() 时如果 logits 在其他 device（如 MPS），
        # 会自动迁移到对应 device（MPS 不支持 CPU Generator）
        self._generator = torch.Generator()
        if self.params.seed is not None:
            self._generator.manual_seed(self.params.seed)
        self._generator_device: torch.device | None = torch.device("cpu")
        # per-request 已生成 token ids，用于 repetition penalty
        self._generated_ids: list[list[int]] | None = None

    def _ensure_generator(self, device: torch.device) -> torch.Generator:
        """确保 Generator 在正确的 device 上（延迟迁移 + 缓存）。

        MPS / CUDA 上的 multinomial 要求 generator 和 tensor 在同一 device，
        首次遇到新 device 时重新创建 Generator 并迁移 seed。
        """
        if self._generator_device == device:
            return self._generator
        # device 变了 → 重新创建并迁移 seed
        seed = self._generator.initial_seed()
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(seed)
        self._generator_device = device
        return self._generator

    def set_generated_ids(self, ids_per_request: list[list[int]]) -> None:
        """注入每个请求已生成的 token id 列表（decode 阶段每步调用）。

        repetition penalty 需要知道哪些 token 已经出现过，才能对它们的 logit
        做惩罚。由 AsyncEngine / batch_generate_loop 在每步 decode 前注入。

        Args:
            ids_per_request: 长度为 B 的列表，每个元素是该请求已生成的 token ids。
                例如 [[1, 5, 23], [7, 12]] 表示 batch 中有 2 个请求，
                第 1 个已生成 3 个 token，第 2 个已生成 2 个。
                传 [] 表示 prefill 阶段（还没有 generated tokens），penalty 不生效。
        """
        self._generated_ids = ids_per_request

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """logits [B, V] -> next_token_ids [B, 1]。

        处理流程：
          temperature=0 → greedy argmax
          temperature>0 → penalty → temperature → top-k → top-p → softmax → sample
        """
        # ── greedy 快速路径 ──
        if self.params.temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # 后续操作在 fp32 下做，避免 bf16/fp16 下 softmax / log 精度不足
        logits = logits.float()

        # ── 1. repetition penalty ──
        # 论文：Ctrl et al., "A Conditional Transformer Language Model for
        # Controllable Generation" (2019). https://arxiv.org/abs/1909.05858
        #
        # 核心思想：对已出现过的 token 的 logit 做惩罚，降低重复生成的概率。
        # 数学公式（对每个已生成的 token v）：
        #   if logit[v] > 0:  logit[v] = logit[v] / penalty   (正数变小)
        #   if logit[v] ≤ 0:  logit[v] = logit[v] * penalty   (负数更负)
        #
        # penalty > 1.0 时降低重复概率，penalty = 1.0 不惩罚。
        # 注意：penalty 作用于 logit（softmax 之前的原始分数），不是概率。
        if self.params.repetition_penalty != 1.0 and self._generated_ids is not None:
            penalty = self.params.repetition_penalty
            for i, ids in enumerate(self._generated_ids):
                if not ids:
                    continue
                # 去重：同一个 token 只惩罚一次
                unique_ids = list(set(ids))
                prev_logits = logits[i, unique_ids]
                # 正 logit 除以 penalty（变小），负 logit 乘以 penalty（变得更负）
                logits[i, unique_ids] = torch.where(
                    prev_logits > 0,
                    prev_logits / penalty,
                    prev_logits * penalty,
                )

        # ── 2. temperature scaling ──
        # 除以温度：T>1 使分布更平滑（探索更多 token），T<1 使分布更锐利（集中高概率 token）
        logits = logits / self.params.temperature

        # ── 3. top-k filtering ──
        # 只保留概率最高的 k 个 token，其余设为 -inf
        if self.params.top_k > 0:
            top_k = min(self.params.top_k, logits.size(-1))
            # kthvalue 找第 k 小的值，比它小的全部 mask 掉
            threshold = torch.kthvalue(logits, logits.size(-1) - top_k + 1, dim=-1).values
            logits = logits.masked_fill(logits < threshold.unsqueeze(-1), float("-inf"))

        # ── 4. top-p (nucleus) filtering ──
        # 论文：Holtzman et al., "The Curious Case of Neural Text Degeneration"
        # (ICLR 2020). https://arxiv.org/abs/1904.09751
        #
        # 核心思想：不像 top-k 固定候选集大小，top-p 动态选择累积概率 ≤ p
        # 的最小 token 集合。概率集中时候选少，概率分散时候选多。
        #
        # 算法步骤：
        #   1. 降序排列 logits
        #   2. 计算 softmax → 累积概率
        #   3. 找到累积概率首次超过 p 的位置，之后的 token 全部 mask 为 -inf
        #   4. 关键细节：保留第一个超过阈值的 token（避免候选集为空）
        #      → 用 cum_probs - sorted_probs > p 而不是 cum_probs > p
        if self.params.top_p < 1.0:
            # 降序排列
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            # 累积概率
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            # 找到累积概率超过 top_p 的位置，这些 token 要移除
            # 但保留第一个超过阈值的 token（避免全部被移除）
            # 例：sorted_probs=[0.5, 0.3, 0.15, 0.05], top_p=0.8
            #   cum_probs=[0.5, 0.8, 0.95, 1.0]
            #   cum_probs - sorted_probs = [0.0, 0.5, 0.8, 0.95]
            #   mask = [False, False, True, True]  → 保留前 2 个
            sorted_mask = cum_probs - sorted_probs > self.params.top_p
            # 把超出范围的 logit 设为 -inf
            sorted_logits[sorted_mask] = float("-inf")
            # 还原到原始顺序
            logits = sorted_logits.scatter(dim=-1, index=sorted_indices, src=sorted_logits)

        # ── 5. softmax → multinomial sample ──
        probs = torch.softmax(logits, dim=-1)
        # multinomial 从每行的概率分布中采样 1 个 token
        # generator 必须与 probs 在同一 device（MPS 不支持 CPU Generator）
        generator = self._ensure_generator(logits.device)
        next_token = torch.multinomial(probs, num_samples=1, generator=generator)
        return next_token
