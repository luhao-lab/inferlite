# M1-T1 RMSNorm

## 元信息
- **任务 ID**: T1
- **里程碑**: M1·P1
- **状态**: ✅ done (2026-06-07, commit `d36b5da`)
- **前置**: T0（实际上 T1 用了硬编码 eps，T0 后回填）
- **实际耗时**: ~1.5h（含环境调试）

## 目标
手写 Qwen3 / LLaMA 系列归一化层，数值上与 `transformers.Qwen3RMSNorm` 完全对齐。

## 产出文件
- `inferlite/model/layers.py::RMSNorm`（27 行含 docstring）
- `tests/unit/test_rmsnorm.py`（12 cases）

## 算法核心
```python
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(input_dtype)
```

## L0 测试结果

| # | 测什么 | Ground truth | 容差 | 结果 |
| --- | --- | --- | --- | --- |
| 1-9 | 3 shape × 3 dtype vs ref | `Qwen3RMSNorm` | fp32 1e-5 / fp16/bf16 1e-3 | ✅ |
| 10 | shape invariant（1D/2D/3D/4D） | — | exact | ✅ |
| 11 | weight 是 nn.Parameter + ones | — | exact | ✅ |
| 12 | eps 默认 1e-6 | — | exact | ✅ |

**12/12 全绿**。

## DoD
- [x] 测试 12/12 绿
- [x] commit `feat(model): RMSNorm + 12 unit tests passing (T1 done)`
- [x] PROGRESS.md 已更新
- [x] docs/2-tasks/README.md 状态 ✅

## 实战教训
1. eps 属性名 → 跟社区一致用 `.eps`，不要自创 `.variance_eps`
2. 必须升 fp32 算 var，否则 fp16/bf16 数值爆炸
3. `.to(input_dtype)` 必须在 `self.weight * x` 之后（先乘 fp32 再降精度）
4. 测试里函数内重复 import 应删除（cosmetic）

## 链接
- 实现: `inferlite/model/layers.py`
- 测试: `tests/unit/test_rmsnorm.py`
- ground truth: `transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm`
