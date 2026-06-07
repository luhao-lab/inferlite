# M1-T0' ModelConfig

## 元信息
- **任务 ID**: T0'
- **里程碑**: M1a
- **状态**: ⬜ pending
- **前置**: T0（包骨架，已 ✅）
- **估时**: 20 min

## 目标
建立 `ModelConfig` dataclass，统一管理 Qwen3 11 个超参；从 `config.json` 反序列化。

## 产出文件
- `inferlite/config.py::ModelConfig`
- `tests/unit/test_config.py`

## 算法核心
```python
# inferlite/config.py
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class ModelConfig:
    # 11 个字段（来自 docs/1-plan/M1.md §2.2）
    hidden_size: int            # H
    num_hidden_layers: int      # N
    num_attention_heads: int
    num_key_value_heads: int    # GQA
    head_dim: int
    intermediate_size: int      # I
    vocab_size: int             # V
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        # TODO: 从 HF config.json 读取并过滤出 11 个字段
        ...

    @classmethod
    def qwen3_0_6b(cls) -> "ModelConfig":
        # TODO: 硬编码 Qwen3-0.6B 超参（便于测试）
        ...
```

## L0 测试清单

| # | 测什么 | 期望 |
| --- | --- | --- |
| 1 | `qwen3_0_6b()` 11 字段全对 | H=1024, N=28, vocab=151936, ... |
| 2 | `from_json(modelscope cache 路径)` 与 hard-coded 一致 | exact eq |
| 3 | `frozen=True` 不允许字段修改 | 触发 FrozenInstanceError |
| 4 | `head_dim == hidden_size / num_attention_heads` 自动校验 | __post_init__ assert |

## DoD
- [ ] 测试 4/4 绿
- [ ] commit `feat(config): ModelConfig dataclass (T0' done)`
- [ ] 回填 `RMSNorm(config.hidden_size, eps=config.rms_norm_eps)` 调用点（暂无，等 T6 一并改）
- [ ] PROGRESS.md / docs/2-tasks/README.md 更新

## 坑（按概率排序）
1. **HF config.json 有几十个字段**，不要全收 —— 按白名单只取 11 个；其余字段未来 M 用到再加
2. **`head_dim` 在某些模型 config.json 不存在** —— 用 `hidden_size // num_attention_heads` 兜底
3. **`tie_word_embeddings` 在 Qwen3-0.6B 为 true，>0.6B 为 false** —— 不要硬编码 True
4. **`dataclass(frozen=True)` 配合 `__post_init__` 校验时**，赋值要用 `object.__setattr__(self, 'x', v)`

## 启动 checklist
- [ ] `docs/1-plan/M1.md` §2.2 11 字段表已读
- [ ] ModelScope 缓存里能找到 `config.json`（`~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B/config.json`）
- [ ] dataclass 基础语法（frozen / __post_init__ / classmethod）确认

## 链接
- 字段来源: `docs/1-plan/M1.md` §2.2
- ground truth: 该 JSON 本身（不需 transformers 对照）
