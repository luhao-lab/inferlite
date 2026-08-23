"""采样策略包。

提供两代采样器：

  - GreedySampler（M2）：纯 argmax，无随机性，适合确定性测试
  - SamplingProcessor（M6）：可配置的采样流水线，支持
    temperature / top-k / top-p / repetition penalty / seed

两者接口一致：__call__(logits: [B, V]) -> [B, 1]，
可以在 batch_generate_loop 和 AsyncEngine 中互换使用。
"""

from inferlite.sampler.greedy import GreedySampler
from inferlite.sampler.sampling import SamplingParams, SamplingProcessor

__all__ = ["GreedySampler", "SamplingParams", "SamplingProcessor"]
