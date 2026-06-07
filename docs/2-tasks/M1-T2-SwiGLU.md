# M1-T2 SwiGLU MLP

## 元信息
- **任务 ID**: T2
- **里程碑**: M1a
- **状态**: ⬜ pending
- **前置**: T0'（或直接用 hidden_size/intermediate_size 硬参，T6 时回填）
- **估时**: 1h

## 目标
手写 Qwen3 的 SwiGLU MLP（门控 + SiLU），与 `transformers.Qwen3MLP` 数值对齐。

## 产出文件
- `inferlite/model/layers.py::SwiGLUMLP`
- `tests/unit/test_mlp.py`

## 背景
经典 MLP: `y = W2 · relu(W1 · x)`（2 个权重 + ReLU）
SwiGLU: `y = W_down · (silu(W_gate · x) ⊙ W_up · x)`（3 个权重 + 门控）

| 维度 | 经典 | SwiGLU |
| --- | --- | --- |
| 权重数 | 2 | **3** |
| 激活 | ReLU | **SiLU**（`x * sigmoid(x)`） |
| 门控 | 无 | 有（gate 路 × up 路） |

Qwen3-0.6B: H=1024, I=3072, **bias=False**

## 算法核心
```python
class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # TODO: 3 个 Linear，全部 bias=False
        # - gate_proj: H -> I
        # - up_proj:   H -> I
        # - down_proj: I -> H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)        # [..., I]
        up   = self.up_proj(x)          # [..., I]
        hidden = F.silu(gate) * up      # element-wise gating
        return self.down_proj(hidden)   # [..., H]
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1-3 | vs `Qwen3MLP`，3 dtype (fp32/fp16/bf16) | `transformers.models.qwen3.modeling_qwen3.Qwen3MLP` | fp32 1e-5 / 其他 5e-3 |
| 4 | bias 全为 None | — | exact |
| 5 | shape invariant (1D/2D/3D) | — | exact |
| 6 | 恰好 3 个 nn.Linear submodule | — | exact |

测试样板见 `docs/1-plan/M1.md` §6.2。

## DoD
- [ ] 测试 6/6 绿
- [ ] commit `feat(model): SwiGLUMLP aligned with Qwen3MLP (T2 done)`
- [ ] PROGRESS.md / docs/2-tasks/README.md 更新

## 坑（按概率排序）
1. `nn.Linear(..., bias=True)` 是默认 → 必须显式 `bias=False`
2. 写成 `F.silu(gate * up)` → 错，应该 `F.silu(gate) * up`（只 gate 路过激活）
3. gate 和 up 顺序写反 → 单测会过（对称），但 weight loading 会错位
4. 自己写 `silu = x * torch.sigmoid(x)` 正确但慢 → 用 `F.silu(...)`
5. 测试用 `Qwen3MLP(cfg)` 需要 `Qwen3Config(mlp_bias=False)`（虽然默认就是 False）

## 启动 checklist
- [ ] T1 RMSNorm 测试仍 12/12 绿（防回归）
- [ ] `docs/1-plan/M1.md` §6.2 SwiGLU 章节已读
- [ ] transformers.models.qwen3.modeling_qwen3.Qwen3MLP 源码已扫一眼

## 链接
- 详细模板: `docs/1-plan/M1.md` §6.2
- ground truth: `transformers.models.qwen3.modeling_qwen3.Qwen3MLP`
