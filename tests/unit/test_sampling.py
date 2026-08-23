"""M6 SamplingParams / SamplingProcessor 单元测试。

覆盖 L0 验证项（全部是纯函数测试，不需要加载模型）：
  1. greedy 快速路径（temperature=0 → argmax）
     → 确保 SamplingProcessor 在 greedy 模式下与 GreedySampler 完全等价
  2. temperature scaling（T>1 更随机，T<1 更集中）
     → 通过统计多次采样的分布来验证温度参数的效果
  3. top-k 过滤（只保留 k 个最高概率 token）
     → 构造特定 logits，验证低概率 token 被正确过滤
  4. top-p 过滤（累积概率 ≤ p 的最小集合）
     → 构造极端 logits，验证 nucleus sampling 的边界行为
  5. repetition penalty（已生成 token 被惩罚）
     → 验证惩罚后 logit 变化导致采样结果改变
  6. seed 可复现（同 seed → 同结果）
     → 两个独立 SamplingProcessor 用相同 seed 应产出相同序列
  7. 接口兼容 GreedySampler（输入 [B,V] 输出 [B,1]）
     → 确保可以直接替换 batch_generate_loop 中的 GreedySampler

测试策略：
  - 所有测试使用手工构造的 logits（不需要真实模型），运行快且确定性高
  - 统计性测试（如温度影响）通过循环多次采样 + set 去重来验证分布特征
  - 每个 test class 对应一个独立的采样策略维度，方便定位问题
"""

import torch

from inferlite.sampler.sampling import SamplingParams, SamplingProcessor


class TestGreedyPath:
    """temperature=0 时退化为 greedy argmax。"""

    def test_argmax_basic(self):
        """greedy 模式返回 logit 最大的 token。"""
        processor = SamplingProcessor(SamplingParams(temperature=0.0))
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])  # V=4, argmax=1
        result = processor(logits)
        assert result.shape == (1, 1)
        assert result.item() == 1

    def test_argmax_batch(self):
        """greedy 模式支持 batch 输入。"""
        processor = SamplingProcessor(SamplingParams(temperature=0.0))
        logits = torch.tensor(
            [
                [1.0, 5.0, 3.0],  # argmax=1
                [9.0, 2.0, 1.0],  # argmax=0
            ]
        )
        result = processor(logits)
        assert result.shape == (2, 1)
        assert result[0].item() == 1
        assert result[1].item() == 0

    def test_greedy_matches_greedy_sampler(self):
        """SamplingProcessor(temperature=0) 与 GreedySampler 结果一致。"""
        from inferlite.sampler.greedy import GreedySampler

        greedy = GreedySampler()
        proc = SamplingProcessor(SamplingParams(temperature=0.0))
        logits = torch.randn(4, 100)
        assert torch.equal(greedy(logits), proc(logits))


class TestTemperature:
    """temperature 对采样分布的影响。"""

    def test_output_shape(self):
        """temperature>0 时输出形状仍为 [B, 1]。"""
        proc = SamplingProcessor(SamplingParams(temperature=1.0, seed=42))
        logits = torch.randn(3, 100)
        result = proc(logits)
        assert result.shape == (3, 1)

    def test_high_temperature_more_random(self):
        """高温采样分布更均匀（不完全集中在 argmax）。"""
        proc = SamplingProcessor(SamplingParams(temperature=10.0, seed=42))
        # 构造一个有明显偏好的 logits
        logits = torch.tensor([[10.0, 9.0, 8.0, 7.0]])
        # 高温下多次采样应该出现非 argmax 的结果
        results = set()
        for seed in range(100):
            proc._generator.manual_seed(seed)
            r = proc(logits.clone())
            results.add(r.item())
        # 高温下应该采样到多个不同的 token
        assert len(results) > 1

    def test_low_temperature_concentrates(self):
        """低温采样集中在 argmax。"""
        proc = SamplingProcessor(SamplingParams(temperature=0.01, seed=42))
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
        # 低温下几乎总是选 argmax
        for seed in range(50):
            proc._generator.manual_seed(seed)
            r = proc(logits.clone())
            assert r.item() == 0


class TestTopK:
    """top-k 过滤。"""

    def test_top_k_filters_low_prob(self):
        """top-k=1 等价于 greedy（只保留最高概率 token）。"""
        proc = SamplingProcessor(SamplingParams(temperature=1.0, top_k=1, seed=42))
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        result = proc(logits)
        assert result.item() == 1  # 只能选 argmax

    def test_top_k_2_two_choices(self):
        """top-k=2 只在最高 2 个 token 中采样。"""
        proc = SamplingProcessor(SamplingParams(temperature=1.0, top_k=2, seed=42))
        logits = torch.tensor([[1.0, 5.0, 4.0, 0.0]])
        results = set()
        for seed in range(100):
            proc._generator.manual_seed(seed)
            r = proc(logits.clone())
            results.add(r.item())
        # 只应该在 {1, 2} 中采样（logit 最高的两个）
        assert results.issubset({1, 2})


class TestTopP:
    """top-p (nucleus) 过滤。"""

    def test_top_p_1_no_filter(self):
        """top_p=1.0 不做过滤。"""
        proc = SamplingProcessor(SamplingParams(temperature=1.0, top_p=1.0, seed=42))
        logits = torch.randn(1, 100)
        result = proc(logits)
        assert result.shape == (1, 1)

    def test_top_p_small_concentrates(self):
        """top_p 很小时集中在少数高概率 token。"""
        proc = SamplingProcessor(SamplingParams(temperature=1.0, top_p=0.01, seed=42))
        logits = torch.tensor([[100.0, 1.0, 1.0, 1.0]])  # token 0 概率极高
        for seed in range(50):
            proc._generator.manual_seed(seed)
            r = proc(logits.clone())
            assert r.item() == 0


class TestRepetitionPenalty:
    """repetition penalty。"""

    def test_penalty_reduces_repeat(self):
        """penalty>1 降低已生成 token 的 logit。"""
        proc = SamplingProcessor(SamplingParams(temperature=0.01, repetition_penalty=10.0, seed=42))
        # token 1 和 2 的 logit 本来最高
        logits = torch.tensor([[1.0, 5.0, 5.0, 1.0]])
        # 不惩罚时应该选 token 1 或 2
        proc.set_generated_ids([])
        r = proc(logits.clone())
        assert r.item() in (1, 2)

        # 惩罚 token 1 和 2 后（penalty=10），logit 变成 0.5，低于 1.0
        proc.set_generated_ids([[1, 2]])
        r = proc(logits.clone())
        assert r.item() in (0, 3)

    def test_penalty_no_effect_without_ids(self):
        """没有 set_generated_ids 时 penalty 不生效。"""
        proc = SamplingProcessor(SamplingParams(temperature=0.0, repetition_penalty=2.0))
        logits = torch.tensor([[1.0, 5.0, 3.0]])
        result = proc(logits)
        assert result.item() == 1  # 仍然是 argmax


class TestSeedReproducibility:
    """seed 可复现测试。"""

    def test_same_seed_same_result(self):
        """相同 seed 产生相同结果。"""
        logits = torch.randn(2, 100)
        proc1 = SamplingProcessor(SamplingParams(temperature=1.0, seed=123))
        proc2 = SamplingProcessor(SamplingParams(temperature=1.0, seed=123))
        r1 = proc1(logits.clone())
        r2 = proc2(logits.clone())
        assert torch.equal(r1, r2)

    def test_different_seed_different_result(self):
        """不同 seed 大概率产生不同结果。"""
        logits = torch.randn(1, 1000)  # 大 vocab 降低碰撞概率
        proc1 = SamplingProcessor(SamplingParams(temperature=1.0, seed=1))
        proc2 = SamplingProcessor(SamplingParams(temperature=1.0, seed=2))
        r1 = proc1(logits.clone())
        r2 = proc2(logits.clone())
        # 大概率不等（1/1000 碰撞概率）
        assert not torch.equal(r1, r2)


class TestInterfaceCompatibility:
    """接口兼容 GreedySampler。"""

    def test_callable_protocol(self):
        """SamplingProcessor 可以被当作 sampler 调用。"""
        proc = SamplingProcessor(SamplingParams(temperature=0.0))
        logits = torch.randn(4, 50)
        result = proc(logits)
        assert result.shape == (4, 1)
        assert result.dtype == torch.long
