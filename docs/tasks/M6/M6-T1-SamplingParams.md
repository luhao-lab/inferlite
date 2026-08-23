# M6-T1 SamplingParams + SamplingProcessor

## 元信息
- **任务 ID**: T1
- **里程碑**: M6
- **状态**: ⬜ pending
- **前置**: 无
- **估时**: 3h

## 目标

**要解决什么问题**：
当前 sampler 只有 `GreedySampler`（argmax），不支持 temperature / top-k / top-p / repetition_penalty 等采样策略。OpenAI API 的请求参数无法生效。

**做完是什么效果**：
```python
params = SamplingParams(temperature=0.7, top_p=0.9, seed=42)
processor = SamplingProcessor(params)
token = processor.sample(logits, generated_ids=[1, 2, 3])
```

**不做什么**（边界）：
- 不改 `GreedySampler`（保持 M1-M5 兼容）
- 不做 beam search
- 不做 frequency/presence penalty（只做 repetition penalty）

**在推理链路中的位置**：
```
logits [B, V]
  ↓
SamplingProcessor.sample(logits, generated_ids)
  ↓
next_token [B, 1]
```

## 产出文件
- `inferlite/sampler/sampling.py` — SamplingParams + SamplingProcessor
- `inferlite/sampler/__init__.py` — 导出
- `tests/unit/test_sampling.py` — 单元测试

## 算法核心

```python
@dataclass(frozen=True)
class SamplingParams:
    """采样参数，对齐 OpenAI API。"""
    temperature: float = 1.0        # 0 = greedy（argmax）
    top_k: int = -1                 # -1 = 不过滤，>0 只保留 top-k
    top_p: float = 1.0             # 1.0 = 不过滤，<1.0 nucleus sampling
    repetition_penalty: float = 1.0 # 1.0 = 不惩罚，>1.0 降权已生成 token
    seed: int | None = None        # None = 不固定 seed

class SamplingProcessor:
    """对 logits 应用采样策略，返回采样的 token id。

    处理流程：
    1. repetition_penalty：对 generated_ids 中的 token 降权
       logits[i] /= penalty if logits[i] > 0 else logits[i] *= penalty
    2. temperature：logits = logits / temperature
    3. top-k：只保留 top-k 个 logits，其余设为 -inf
    4. top-p：按概率降序累积，保留累积概率 >= top_p 的最小集合
    5. softmax → multinomial 采样
    6. temperature=0 退化为 argmax（不调 multinomial）
    """

    def __init__(self, params: SamplingParams):
        self.params = params
        self._generator = None
        if params.seed is not None:
            self._generator = torch.Generator().manual_seed(params.seed)

    def sample(self, logits: torch.Tensor,
               generated_ids: list[int] | None = None) -> torch.Tensor:
        """logits [B, V] → next_token [B, 1]"""
        # temperature=0 → argmax（greedy 退化）
        # repetition_penalty（fp32 下操作）
        # temperature scaling
        # top-k filtering
        # top-p (nucleus) filtering
        # softmax + multinomial
        ...
```

### 关键实现细节

1. **repetition_penalty**：
   ```python
   # 对已生成 token 的 logit 做惩罚
   for token_id in generated_ids:
       if logits[token_id] > 0:
           logits[token_id] /= penalty
       else:
           logits[token_id] *= penalty  # 负数乘 >1 变得更负
   ```

2. **top-p (nucleus) filtering**：
   ```python
   sorted_logits, sorted_indices = torch.sort(logits, descending=True)
   cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
   # 移除累积概率 > top_p 的 token（保留至少 1 个）
   sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
   sorted_logits[sorted_mask] = float('-inf')
   # 恢复原顺序
   logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
   ```

3. **seed 可复现**：`torch.Generator` 实例，每次 sample 共享同一个 generator（状态递进）

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
|---|--------|-------------|------|
| 1 | temperature=0 等价 argmax | `torch.argmax(logits)` | exact |
| 2 | seed 可复现 | 同 seed 两次采样 | exact |
| 3 | top-k=1 等价 argmax | `torch.argmax(logits)` | exact |
| 4 | top-k 过滤 | 采样结果在 top-k 范围内 | exact |
| 5 | top-p=0.1 只采高概率 | 结果在 nucleus set 内 | exact |
| 6 | repetition_penalty 降权 | 已生成 token 的 logit 被修改 | exact |
| 7 | temperature > 1 增加随机性 | 统计分布更均匀 | statistical |
| 8 | 边界值 | T=0.01, T=2.0, top_k=1, top_p=0.01 | no crash |
| 9 | 多维 logits | [B, V] 正常工作 | shape correct |
| 10 | generated_ids=None | 跳过 repetition penalty | no crash |
| 11 | greedy 与 GreedySampler 一致 | `GreedySampler(logits)` | exact |
| 12 | fp32 数值安全 | repetition_penalty 在 fp32 下 | atol 1e-5 |

## DoD
- [ ] 测试 12/12 全绿
- [ ] commit `feat(sampler): add SamplingParams + SamplingProcessor (T1 done)`
- [ ] `sampler/__init__.py` 导出 SamplingParams + SamplingProcessor
- [ ] PROGRESS.md 更新

## 坑（按概率排序）
1. top-p 的"至少保留 1 个 token"边界处理
2. repetition_penalty 对负 logit 的处理方向（乘 vs 除）
3. `torch.Generator` 跨设备问题（MPS 上 generator 可能不支持）
4. temperature 极小值（0.01）的数值稳定性
