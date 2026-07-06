# M2-T1 KVCache 数据结构

## 元信息
- **任务 ID**: M2-T1
- **里程碑**: M2（KV Cache）
- **状态**: ✅ done
- **前置**: M1 全部完成（tag `m1/naive-forward`）
- **估时**: 1h

## 目标

**要解决什么问题**：
M1 的 generate loop 每步都对所有历史 token 重算 K/V，复杂度 O(T²)。
本卡定义"放 KV 的盒子"——KV Cache 数据结构，为后续 T2（Attention 读写）和 T4（generate loop 拆分）打好地基。

**做完是什么效果**：
```python
cache = KVCache.from_config(ModelConfig.qwen3_0_6b(), batch_size=1,
                             max_seq_len=1024, dtype=torch.float32, device="cpu")
# 成功创建：28 层 × 2（K+V）个 [1, 8, 1024, 128] tensor，全部预分配到 cpu
assert len(cache.layers) == 28
assert cache.layers[0].k.shape == (1, 8, 1024, 128)
assert cache.cur_len == 0
```

**不做什么**（边界）：
本卡只是数据结构（造盒子），不写读写逻辑。
读写（切片写入 K/V、切片读取有效历史）在 T2（Attention）实现；
cur_len 的推进在 T4（generate loop）实现。

**在推理链路中的位置**：
```
generate()                        ← T4 负责
    ├── KVCache.from_config()     ← 本卡：一次性预分配所有层的 K/V tensor
    │
    ├── Prefill: model.forward(input_ids, kv_cache=cache)
    │       └── 每层 Attention 拿到 cache.layers[i]（T2 负责读写）
    │
    └── Decode loop（每步）
            └── 每层 Attention 读 cache.layers[i]（T2 负责）
                cur_len += 1     ← T4 负责推进
```

## 产出文件
- `inferlite/model/kv_cache.py` — `LayerKVCache` + `KVCache`
- `tests/unit/test_kv_cache.py`

## 参考代码
- 设计文档 §3.1、§4 ADR-01：`inferlite/docs/m2-kv-cache-design.md`
- transformers `StaticCache`：https://github.com/huggingface/transformers/blob/main/src/transformers/cache_utils.py

## 算法核心

```python
# inferlite/model/kv_cache.py（新建）
from dataclasses import dataclass
import torch

@dataclass
class LayerKVCache:
    k: torch.Tensor  # [B, n_kv_heads, max_seq_len, head_dim]
    v: torch.Tensor  # [B, n_kv_heads, max_seq_len, head_dim]

class KVCache:
    def __init__(self, layers: list[LayerKVCache]) -> None:
        self.layers = layers
        self.cur_len: int = 0   # 已写入的有效 token 数，所有层共享

    @classmethod
    def from_config(cls, config, batch_size: int, max_seq_len: int,
                    dtype: torch.dtype, device) -> "KVCache":
        # TODO：按 config 分配所有层的 k/v tensor
        ...

    def reset(self) -> None:
        self.cur_len = 0        # tensor 不清零，下次 prefill 会覆盖写入
```

**关键设计点**：
- `cur_len` 是唯一事实源，在 generate loop 里显式更新（见 ADR-02）
- `reset()` 只重置 `cur_len`，不清零 tensor（节约时间，prefill 会覆盖）
- `from_config` 接收 `ModelConfig`，自动读取 `num_hidden_layers`、`num_key_value_heads`、`head_dim`

## L0 测试清单

| # | 测什么 | Ground truth | 容差 |
| --- | --- | --- | --- |
| 1 | `from_config` 分配的 `k/v` shape | `[B, n_kv, max_seq_len, D]` | exact |
| 2 | `len(kv_cache.layers)` == `num_hidden_layers` | `config.num_hidden_layers` | exact |
| 3 | `cur_len` 初始为 0 | 0 | exact |
| 4 | `reset()` 后 `cur_len` 归零 | 0 | exact |
| 5 | `cur_len > max_seq_len` 时写入触发 `IndexError` | IndexError | — |
| 6 | device 和 dtype 与参数一致 | 构造参数 | exact |

## DoD
- [x] `tests/unit/test_kv_cache.py` 全绿
- [x] `KVCache.from_config(ModelConfig.qwen3_0_6b(), B=1, max_seq_len=1024, dtype=torch.float32, device="cpu")` 分配正确
- [x] `uv run pytest tests/unit/test_kv_cache.py -q` 通过
- [x] commit `feat(model): add KVCache data structure (M2-T1)` (`474c04f`)
- [x] 任务状态改 ✅

## 坑（按概率排序）
1. **`cur_len` 不要放在 `LayerKVCache` 里**：各层共享同一个 `cur_len`，放在 `KVCache` 层管理，避免多层不同步的 bug。
2. **reset 不清零 tensor**：只重置 `cur_len = 0`，prefill 阶段的写入会覆盖旧值。
3. **`from_config` 的 device 参数**：tensor 要分配到正确 device，否则后续 attention 有 device mismatch。
4. **越界检查时机**：越界应在 Attention 写入时（切片赋值）自然触发 `IndexError`，不需要在 `from_config` 里做额外检查。
