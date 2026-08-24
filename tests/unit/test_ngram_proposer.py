"""M7-T1 NgramProposer 单元测试。

8 个测试用例覆盖：
1. 完美匹配（query 在 context 中出现）
2. 无匹配（query 从未出现）
3. 部分匹配（只有短 n-gram 匹配）
4. 边界情况（context 太短）
5. 避免自匹配（query 不能匹配自己）
6. draft 数量限制（匹配后 token 不够）
7. 多匹配取最长（优先长 n-gram）
8. 中文 token 序列（验证 token 无关性）
"""

import pytest

from inferlite.spec.ngram_proposer import NgramProposer


class TestNgramProposer:
    """NgramProposer 的 8 个测试用例。"""

    def test_1_perfect_match(self):
        """测试 1：完美匹配（query 在 context 中出现）。

        context = [1, 2, 3, 4, 5, 1, 2, 3]
        query = [1, 2, 3]（末尾 3 个 token）
        匹配位置 = 0（context[0:3] == [1, 2, 3]）
        提取 context[3:8] = [4, 5, 1, 2, 3]，但截断到 num_draft_tokens=5

        预期：[4, 5]（匹配位置后面只有 2 个 token）
        """
        proposer = NgramProposer(min_n=1, max_n=4, num_draft_tokens=5)
        context = [1, 2, 3, 4, 5, 1, 2, 3]
        draft = proposer.propose(context)

        # 匹配到 context[0:3] = [1, 2, 3]，提取 context[3:8]
        # 但 context[3:8] = [4, 5, 1, 2, 3]，截断到 num_draft_tokens=5
        # 实际应该是 [4, 5, 1, 2, 3]
        # 等等，让我重新算：
        # - query = context[-3:] = [1, 2, 3]
        # - 匹配位置 i=0（context[0:3] == [1, 2, 3]）
        # - start = i + n = 0 + 3 = 3
        # - end = min(3 + 5, 8) = min(8, 8) = 8
        # - draft = context[3:8] = [4, 5, 1, 2, 3]

        assert draft == [4, 5, 1, 2, 3]

    def test_2_no_match(self):
        """测试 2：无匹配（query 从未出现）。

        context = [1, 2, 3, 4, 5]
        query = [3, 4, 5]（末尾 3 个 token）
        context[:-3] = [1, 2]，没有匹配

        尝试更短的 n=2：query = [4, 5]，context[:-2] = [1, 2, 3]，没有匹配
        尝试更短的 n=1：query = [5]，context[:-1] = [1, 2, 3, 4]，没有匹配

        预期：[]（无匹配）
        """
        proposer = NgramProposer(min_n=1, max_n=4, num_draft_tokens=5)
        context = [1, 2, 3, 4, 5]
        draft = proposer.propose(context)

        assert draft == []

    def test_3_partial_match_short_ngram(self):
        """测试 3：部分匹配（只有短 n-gram 匹配）。

        context = [1, 2, 3, 4, 5, 2, 3]
        query = [2, 3]（末尾 2 个 token）
        匹配位置 = 1（context[1:3] == [2, 3]）
        提取 context[3:8] = [4, 5, 2, 3]

        尝试 n=4, 3 时没有匹配，所以用 n=2

        预期：[4, 5, 2, 3]
        """
        proposer = NgramProposer(min_n=1, max_n=4, num_draft_tokens=5)
        context = [1, 2, 3, 4, 5, 2, 3]
        draft = proposer.propose(context)

        # n=4: query=[3,4,5,2], context[:-4]=[1,2,3], 无匹配
        # n=3: query=[5,2,3], context[:-3]=[1,2,3,4], 无匹配
        # n=2: query=[2,3], context[:-2]=[1,2,3,4,5], 匹配位置 i=1
        #   start=1+2=3, end=min(3+5, 7)=7, draft=context[3:7]=[4,5,2,3]
        assert draft == [4, 5, 2, 3]

    def test_4_context_too_short(self):
        """测试 4：边界情况（context 太短）。

        context = [1, 2]
        min_n = 3，context 长度 < min_n

        预期：[]（无法匹配）
        """
        proposer = NgramProposer(min_n=3, max_n=4, num_draft_tokens=5)
        context = [1, 2]
        draft = proposer.propose(context)

        assert draft == []

    def test_5_avoid_self_matching(self):
        """测试 5：避免自匹配（query 不能匹配自己）。

        context = [1, 2, 3]
        query = [1, 2, 3]（末尾 3 个 token）

        如果匹配到 context[-3:]（自己），会提取 context[0:] = [1, 2, 3]，
        这是已经生成过的 token，不是新的 draft。

        算法在 context[:-3] = [] 中找匹配，找不到。

        预期：[]（避免自匹配）
        """
        proposer = NgramProposer(min_n=1, max_n=4, num_draft_tokens=5)
        context = [1, 2, 3]
        draft = proposer.propose(context)

        # n=4: context 长度 < 4，跳过
        # n=3: query=[1,2,3], context[:-3]=[], 无匹配
        # n=2: query=[2,3], context[:-2]=[1], 无匹配
        # n=1: query=[3], context[:-1]=[1,2], 无匹配
        assert draft == []

    def test_6_draft_count_limit(self):
        """测试 6：draft 数量限制（匹配后 token 不够）。

        context = [1, 2, 3, 4, 1, 2, 3]
        query = [1, 2, 3]（末尾 3 个 token）
        匹配位置 = 0（context[0:3] == [1, 2, 3]）
        提取 context[3:8] = [4, 1, 2, 3]（只有 4 个 token，不够 num_draft_tokens=5）

        预期：[4, 1, 2, 3]（返回所有可用 token）
        """
        proposer = NgramProposer(min_n=1, max_n=4, num_draft_tokens=5)
        context = [1, 2, 3, 4, 1, 2, 3]
        draft = proposer.propose(context)

        # 匹配位置 i=0, start=3, end=min(3+5, 7)=7
        # draft = context[3:7] = [4, 1, 2, 3]
        assert draft == [4, 1, 2, 3]

    def test_7_prefer_longest_ngram(self):
        """测试 7：多匹配取最长（优先长 n-gram）。

        context = [1, 2, 3, 4, 5, 6, 1, 2, 3]
        query = [1, 2, 3]（末尾 3 个 token）

        n=3 时匹配位置 i=0（context[0:3] == [1, 2, 3]）
        n=2 时也能匹配（context[0:2] == [1, 2]），但优先用 n=3

        预期：用 n=3 匹配，提取 context[3:8] = [4, 5, 6, 1, 2]
        """
        proposer = NgramProposer(min_n=1, max_n=4, num_draft_tokens=5)
        context = [1, 2, 3, 4, 5, 6, 1, 2, 3]
        draft = proposer.propose(context)

        # n=4: query=[6,1,2,3], context[:-4]=[1,2,3,4,5], 无匹配
        # n=3: query=[1,2,3], context[:-3]=[1,2,3,4,5,6], 匹配位置 i=0
        #   start=0+3=3, end=min(3+5, 9)=8, draft=context[3:8]=[4,5,6,1,2]
        assert draft == [4, 5, 6, 1, 2]

    def test_8_arbitrary_token_ids(self):
        """测试 8：中文 token 序列（验证 token 无关性）。

        NgramProposer 只关心 token id，不关心实际内容。
        用任意整数序列测试，确保算法对 token 无关。

        context = [100, 200, 300, 400, 100, 200, 300]
        query = [100, 200, 300]
        匹配位置 = 0
        提取 context[3:8] = [400, 100, 200, 300]

        预期：[400, 100, 200, 300]
        """
        proposer = NgramProposer(min_n=1, max_n=4, num_draft_tokens=5)
        context = [100, 200, 300, 400, 100, 200, 300]
        draft = proposer.propose(context)

        # 匹配位置 i=0, start=3, end=min(3+5, 7)=7
        # draft = context[3:7] = [400, 100, 200, 300]
        assert draft == [400, 100, 200, 300]

    def test_parameter_validation(self):
        """参数验证测试（额外）。

        确保 min_n >= 1, max_n >= min_n, num_draft_tokens >= 1
        """
        # min_n < 1 应该报错
        with pytest.raises(AssertionError):
            NgramProposer(min_n=0, max_n=4, num_draft_tokens=5)

        # max_n < min_n 应该报错
        with pytest.raises(AssertionError):
            NgramProposer(min_n=3, max_n=2, num_draft_tokens=5)

        # num_draft_tokens < 1 应该报错
        with pytest.raises(AssertionError):
            NgramProposer(min_n=1, max_n=4, num_draft_tokens=0)

    def test_multiple_matches_first_wins(self):
        """测试 9：多匹配取第一个（而非最长）。

        context = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        query = [1, 2, 3]

        n=4 时：query=[3,1,2,3]，匹配位置 i=2（context[2:6]=[3,1,2,3]）
        n=3 时：query=[1,2,3]，匹配位置 i=0（context[0:3]=[1,2,3]）

        但算法优先用长 n-gram，所以用 n=4 的匹配

        预期：用 n=4 匹配，提取 context[6:9] = [1, 2, 3]
        """
        proposer = NgramProposer(min_n=1, max_n=4, num_draft_tokens=5)
        context = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        draft = proposer.propose(context)

        # n=4: query=[3,1,2,3], 匹配位置 i=2, start=6, end=9, draft=[1,2,3]
        assert draft == [1, 2, 3]
