"""M6 SamplingProcessor 单元测试。

覆盖：
- SamplingParams 默认值 + is_greedy 判定
- Greedy 快速路径（temperature=0, repetition_penalty=1）
- Temperature scaling
- Top-k 过滤
- Top-p (nucleus) 过滤
- Repetition penalty
- 综合采样流程
"""

import pytest
import torch

from inferlite.sampler.sampling import SamplingParams, SamplingProcessor


class TestSamplingParams:
    """SamplingParams 数据类测试。"""

    def test_defaults(self):
        p = SamplingParams()
        assert p.temperature == 0.0
        assert p.top_k == 0
        assert p.top_p == 1.0
        assert p.repetition_penalty == 1.0

    def test_is_greedy_default(self):
        assert SamplingParams().is_greedy is True

    def test_is_greedy_with_temperature(self):
        assert SamplingParams(temperature=0.7).is_greedy is False

    def test_is_greedy_with_repetition_penalty(self):
        assert SamplingParams(repetition_penalty=1.1).is_greedy is False

    def test_frozen(self):
        p = SamplingParams()
        with pytest.raises(AttributeError):
            p.temperature = 1.0  # type: ignore[misc]


class TestGreedyPath:
    """纯 greedy 快速路径。"""

    def test_greedy_argmax(self):
        logits = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
        processor = SamplingProcessor(SamplingParams())  # temperature=0, no penalty
        result = processor(logits)
        assert result.shape == (1, 1)
        assert result.item() == 1  # argmax of [1, 3, 2, 0.5] = index 1

    def test_greedy_batch(self):
        logits = torch.tensor(
            [
                [1.0, 5.0, 2.0],
                [3.0, 1.0, 4.0],
            ]
        )
        processor = SamplingProcessor()
        result = processor(logits)
        assert result.shape == (2, 1)
        assert result[0].item() == 1
        assert result[1].item() == 2


class TestTemperature:
    """Temperature scaling 测试。"""

    def test_high_temperature_more_uniform(self):
        """高温度让分布更平坦，低概率 token 有更大机会被选中。"""
        torch.manual_seed(42)
        logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])  # 第一个 token 概率很高
        processor = SamplingProcessor(SamplingParams(temperature=100.0))

        # 高温度下，分布趋近均匀，多次采样应该能看到非 argmax 的 token
        results = set()
        for _ in range(100):
            r = processor(logits.clone())
            results.add(r.item())
        assert len(results) > 1  # 至少采样到 2 种不同 token

    def test_low_temperature_near_greedy(self):
        """极低温度接近 greedy。"""
        logits = torch.tensor([[1.0, 10.0, 2.0]])
        processor = SamplingProcessor(SamplingParams(temperature=0.01))

        for _ in range(20):
            r = processor(logits.clone())
            assert r.item() == 1  # 几乎总是选 argmax

    def test_temperature_zero_is_greedy(self):
        """temperature=0 走 argmax 路径。"""
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        processor = SamplingProcessor(SamplingParams(temperature=0.0))
        assert processor(logits).item() == 1


class TestTopK:
    """Top-k 过滤测试。"""

    def test_top_k_1_is_greedy(self):
        """top_k=1 等价于 greedy（只有一个候选）。"""
        torch.manual_seed(42)
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
        processor = SamplingProcessor(SamplingParams(temperature=1.0, top_k=1))
        assert processor(logits).item() == 1

    def test_top_k_filters_low_logits(self):
        """top_k=2 只保留前 2 个 token。"""
        torch.manual_seed(42)
        logits = torch.tensor([[10.0, 9.0, 1.0, 0.5]])
        processor = SamplingProcessor(SamplingParams(temperature=1.0, top_k=2))
        results = set()
        for _ in range(100):
            r = processor(logits.clone())
            results.add(r.item())
        # 只能采样到 index 0 或 1
        assert results <= {0, 1}

    def test_top_k_larger_than_vocab(self):
        """top_k 大于 vocab 大小时不过滤。"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        processor = SamplingProcessor(SamplingParams(temperature=1.0, top_k=100))
        # 不应报错
        result = processor(logits)
        assert result.shape == (1, 1)


class TestTopP:
    """Top-p (nucleus) 过滤测试。"""

    def test_top_p_1_no_filter(self):
        """top_p=1.0 不过滤任何 token。"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        processor = SamplingProcessor(SamplingParams(temperature=1.0, top_p=1.0))
        # 所有 token 都可能被选中
        results = set()
        torch.manual_seed(42)
        for _ in range(100):
            r = processor(logits.clone())
            results.add(r.item())
        assert len(results) >= 2

    def test_top_p_small_restricts(self):
        """很小的 top_p 会大幅限制候选集。"""
        torch.manual_seed(42)
        logits = torch.tensor([[10.0, 0.1, 0.1, 0.1]])  # 第一个 token 概率极高
        processor = SamplingProcessor(SamplingParams(temperature=1.0, top_p=0.5))
        for _ in range(50):
            r = processor(logits.clone())
            assert r.item() == 0  # 只能选到第一个 token


class TestRepetitionPenalty:
    """Repetition penalty 测试。"""

    def test_no_penalty(self):
        """penalty=1.0 不改变 logits。"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        input_ids = torch.tensor([[0, 1]])
        processor = SamplingProcessor(SamplingParams(temperature=0.0, repetition_penalty=1.0))
        assert processor(logits, input_ids).item() == 2  # argmax 不变

    def test_penalty_reduces_repeated(self):
        """repetition_penalty 降低已出现 token 的选择概率。"""
        # token 2 的 logit 本来最高，但如果被惩罚足够大，可能被 token 1 超过
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        input_ids = torch.tensor([[2]])  # token 2 出现过
        processor = SamplingProcessor(SamplingParams(temperature=0.0, repetition_penalty=2.0))
        # 惩罚后: token 2 → 3.0/2.0 = 1.5, token 1 → 2.0 (不变)
        # argmax 应该变成 token 1
        assert processor(logits, input_ids).item() == 1

    def test_penalty_negative_logits(self):
        """负 logit 的惩罚是乘以 penalty（绝对值变大）。"""
        logits = torch.tensor([[-1.0, -2.0, 3.0]])
        input_ids = torch.tensor([[0]])  # token 0 出现过
        result = SamplingProcessor._apply_repetition_penalty(logits.clone(), input_ids, 2.0)
        assert result[0, 0].item() == -2.0  # -1.0 * 2.0 = -2.0
        assert result[0, 1].item() == -2.0  # 不变
        assert result[0, 2].item() == 3.0  # 不变


class TestCombined:
    """综合采样流程测试。"""

    def test_temperature_topk_repetition(self):
        """temperature + top_k + repetition_penalty 组合。"""
        torch.manual_seed(42)
        logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        input_ids = torch.tensor([[0]])  # token 0 出现过
        params = SamplingParams(temperature=1.0, top_k=3, repetition_penalty=1.5)
        processor = SamplingProcessor(params)

        results = set()
        for _ in range(100):
            r = processor(logits.clone(), input_ids.clone())
            results.add(r.item())
        # token 0 被惩罚，top_k=3 只保留前 3
        # 候选集应该是 {0, 1, 2} 中惩罚后的 top 3
        assert results <= {0, 1, 2}

    def test_no_input_ids_skips_penalty(self):
        """input_ids=None 时跳过 repetition penalty。"""
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        processor = SamplingProcessor(SamplingParams(temperature=0.0, repetition_penalty=2.0))
        # 无 input_ids，repetition penalty 不生效，走 greedy
        assert processor(logits, None).item() == 1
