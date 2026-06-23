# M1-T0 ModelConfig

## 元信息
- **任务 ID**: T0
- **里程碑**: M1·P1
- **状态**: ✅ done
- **前置**: M0（包骨架，已 ✅）
- **估时**: 20 min

## 目标
建立 `ModelConfig` dataclass，统一管理 Qwen3 11 个超参；从 `config.json` 反序列化。

## 产出文件
- `inferlite/config.py::ModelConfig`
- `tests/unit/test_config.py`

## 算法核心
```python
# inferlite/config.py
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:
    # 11 个字段（来自 docs/kb/knowledge.md §Qwen3 Tech Report §4）
    hidden_size: int            # H
    num_hidden_layers: int      # N
    num_attention_heads: int    # n_q
    num_key_value_heads: int    # n_kv (GQA)
    head_dim: int               # d, 独立参数，不要推导成 H/n_q
    intermediate_size: int      # I
    vocab_size: int             # V
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        # fail fast: 构造期拒绝明显非法的 config
        ...

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        # 从 HF config.json 读取，按白名单过滤出 M1 用到的 11 个字段
        ...

    @classmethod
    def qwen3_0_6b(cls) -> "ModelConfig":
        # 硬编码 Qwen3-0.6B 超参，作为 disk-independent ground truth
        ...
```

## L0 测试清单

| # | 测什么 | 期望 | 状态 |
| --- | --- | --- | --- |
| 1 | `qwen3_0_6b()` 11 字段全对 | H=1024, N=28, vocab=151936, ... | ✅ |
| 2 | `from_json(tmp config.json)` 与 hard-coded 一致 | exact eq；白名单外字段被过滤 | ✅ |
| 3 | `from_json()` 缺 `head_dim` 时兜底 | `hidden_size // num_attention_heads` | ✅ |
| 4 | `frozen=True` 不允许字段修改 | 触发 FrozenInstanceError | ✅ |
| 5 | `num_attention_heads % num_key_value_heads == 0` (GQA 合法) | __post_init__ assert，错误信息含 GQA | ✅ |

## DoD
- [x] 测试 5/5 绿
- [x] commit `feat(config): ModelConfig dataclass (T0 done)`
- [x] 回填 `RMSNorm(config.hidden_size, eps=config.rms_norm_eps)` 调用点（暂无，等 T6 一并改）
- [x] PROGRESS.md / docs/tasks/README.md 更新

## 坑（按概率排序）
1. **HF config.json 有几十个字段**，不要全收 —— 按白名单只取 11 个；其余字段未来 M 用到再加
2. **`head_dim != hidden_size / num_attention_heads`**（严重坑！）—— Qwen3-0.6B `head_dim=128` 但 `1024/16=64`，**不要写这个 invariant**；GQA/MQA 里 head_dim 是独立参数
3. **`head_dim` 在某些模型 config.json 不存在** —— 用 `hidden_size // num_attention_heads` 兜底，但**优先**从 JSON 读
4. **`tie_word_embeddings` 在 Qwen3-0.6B 为 true，>0.6B 为 false** —— 不要硬编码 True
5. **`dataclass(frozen=True)` 配合 `__post_init__` 校验时**，赋值要用 `object.__setattr__(self, 'x', v)`
6. **`rope_theta` JSON 里是 int (`1000000`)** —— dataclass 字段用 `float`，反序列化时 cast，避免后续 RoPE 计算 dtype 错位
7. **Transformers `Qwen3Config` 不是 0.6B 具体值** —— `Qwen3Config()` 默认值不等于 `Qwen3-0.6B/config.json`；字段定义看前者，具体数值看后者
8. **annotated tag 不能当 GitHub blob commit 用** —— 固定源码链接时要用 `refs/tags/vX.Y.Z^{}` 的 peeled commit，或在 GitHub 按 `y`

## 启动 checklist
- [x] `docs/kb/knowledge.md` Qwen3-0.6B 架构精读已读
- [x] ModelScope 缓存里能找到 `config.json`（`~/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B/config.json`）
- [x] dataclass 基础语法（frozen / __post_init__ / classmethod）确认

## 完成总结

### 产出
- 新增 `inferlite/config.py`
  - `ModelConfig`：Qwen3-0.6B 的 11 个核心超参
  - `from_json()`：读取 HF/ModelScope `config.json`，白名单过滤，`head_dim` 兼容兜底，`rope_theta` cast float
  - `qwen3_0_6b()`：硬编码 0.6B ground truth，便于单测不依赖磁盘缓存
- 新增 `tests/unit/test_config.py`
  - 5 个 L0 单测，覆盖正常路径、异常路径、不可变契约与 GQA invariant

### 验证
```bash
uv run pytest tests/unit/test_config.py -q  # 5 passed
make test                                  # 17 passed
bash scripts/doctor.sh                     # 9/9 PASS
```

### 关键学习
1. `ModelConfig` 的本质是**模型超参合同**，不是 `nn.Module`，不参与 forward。
2. `dataclass(frozen=True)` 适合存不可变超参：少样板代码、可按字段比较、构造后禁止误改。
3. `__post_init__` 是 dataclass 自动初始化后的安全检查点，用来 fail fast。
4. `@classmethod` 适合写 `from_json()` / `qwen3_0_6b()` 这类“还没有对象时创建对象”的工厂。
5. GQA 中 `num_key_value_heads` 减少的是 KV 组数，不改变每个 head 的维度；`head_dim` 优先读 JSON，不能随意推导。

## 链接
- 字段来源: `docs/kb/knowledge.md` → `Qwen3 Tech Report` → `ModelConfig 11 字段`
- Transformers `Qwen3Config` 固定源码: https://github.com/huggingface/transformers/blob/0dad7b822255a0ae261ec45ae937371e859ffd1a/src/transformers/models/qwen3/configuration_qwen3.py
- Qwen3-0.6B `config.json`: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
