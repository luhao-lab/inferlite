# M1-TX <任务名>

> 任务卡模板。复制改名为 `M1-TX-<short-name>.md`。

## 元信息
- **任务 ID**: TX
- **里程碑**: M1·P1 / M1·P2
- **状态**: ⬜ pending / 🟡 in-progress / ✅ done
- **前置**: TY, TZ
- **估时**: Nh

## 目标
一句话说清这张卡产出什么。

## 产出文件
- `inferlite/.../xxx.py::ClassName`
- `tests/unit/test_xxx.py`

## 算法核心
公式 + 代码骨架（TODO 留给作者）：

```python
class ClassName(nn.Module):
    def __init__(self, ...):
        # TODO
        ...
    def forward(self, x):
        # TODO
        ...
```

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | ... | `transformers.Qwen3X` | 1e-5 / 5e-3 |

## DoD
- [ ] 测试 N/N 全绿
- [ ] commit `feat(...): ... (TX done)`
- [ ] PROGRESS.md 更新
- [ ] `README.md` 当前进度同步（若本任务改变首页可见进度）
- [ ] docs/2-tasks/README.md 状态改 ✅

## 坑（按概率排序）
1. ...
2. ...
