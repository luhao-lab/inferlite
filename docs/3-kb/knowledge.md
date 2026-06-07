# Knowledge Cards

> 项目用到的论文 / 库 / 概念 / 工具，原子卡片，可被任务卡/lessons 引用。
> 单文件多 H2，按主题查找，搜索友好。
> 新增卡片直接在对应章节末尾追加。

---

## 📊 索引摘要

> 自动维护：每次新增/删除卡片时同步更新本段。最后更新：2026-06-07。

**总览**：4 类共 **20 张** knowledge 卡片 + 3 条 lessons + 3 个 ADR。

| 章节 | 卡片数 | 列表（点击跳转） |
| --- | --- | --- |
| **Papers** | 5 | [RMSNorm](#rmsnorm-zhang-sennrich-neurips-2019) · [Qwen3 Tech Report](#qwen3-tech-report) · [SwiGLU](#swiglu-shazeer-2020) · [RoPE](#rope-su-et-al-2021) · [GQA](#gqa-ainslie-et-al-2023) |
| **Libraries** | 5 | [transformers Qwen3 模块](#transformers-qwen3-ground-truth) · [transformers 5 核心对象](#transformers-hf) · [pytest](#pytest-api) · [modelscope](#modelscope-snapshot_download) · [huggingface_hub 1.x](#huggingface_hub-1x) |
| **Concepts** | 5 | [upcast fp32](#upcast-to-fp32) · [tie_word_embeddings](#tie_word_embeddings) · [形状速查](#shape-cheatsheet) · [推理上下文](#inference-context) · [Factory pattern](#factory-pattern) |
| **Tools** | 5 | [uv](#uvpython) · [make](#make) · [ruff](#rufflint-format) · [pre-commit](#pre-commitcommit) · [pytest-mark / CI](#pytest-mark-ci-matrix) |

**已沉淀的 lessons**（[详见 lessons.md](./lessons.md)）：

- **L1** RMSNorm 必须 upcast fp32 算方差（→ Concepts#upcast）
- **L2** 国内拉模型用 ModelScope 替代 HF mirror（→ Libraries#modelscope, #huggingface_hub）
- **L3** 地基 vs 算法是两个频道，不要混着切（→ 协作节奏）

**已落地的 ADR**（[详见 decisions.md](./decisions.md)）：

- **ADR-001** spec-driven 工作流 + 双文件知识库
- **ADR-002** 知识库与代码同仓（R1 重构）
- **ADR-003** 分组目录 + MkDocs Material 可视化

**任务卡进度**（[详见 PROGRESS](../1-plan/PROGRESS.md)）：M1-T1 RMSNorm ✅ · T0' ModelConfig 🟡 · T2-T6 ⬜

**已知缺口**（开工时回填）：

- ⚠️ `Concepts#数值对齐策略` — fp32 baseline / bf16 实战 / 三层验证三件套（M1 横跨 T1-T6 的元概念，T0' 完成时回填）
- ⚠️ `Concepts#KV Cache 结构` — 等 M2 任务卡触发
- ⚠️ `Concepts#Continuous Batching 调度` — 等 M3 任务卡触发

---

## Papers

### RMSNorm (Zhang & Sennrich, NeurIPS 2019)

**一句话**：LayerNorm 的简化版（去中心化 + 去偏置 β），LLaMA/Qwen/Mistral 标配。

**公式**：

LayerNorm:  `(x - μ) / σ · γ + β`（2 stats + 2 params）

RMSNorm:    `x / sqrt(mean(x²) + ε) · γ`（1 stat + 1 param）

**关键论点**：
1. 去中心化对训练几乎不影响（"激活分布形状不变"）
2. 参数减半（无 β）
3. 速度比 LN 快 7%-64%（看 framework）
4. NLU/MT 任务上效果持平甚至略好

**本项目对应**：
- 文件：`inferlite/model/layers.py::RMSNorm` (T1)
- ε：Qwen3 用 1e-6（LLaMA-2 同；LLaMA-1 是 1e-5）
- 实现注意：必须 upcast fp32 算 var（见下方 Concepts）

**外部参考**：
- 论文：https://arxiv.org/abs/1910.07467
- transformers：`transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm`
- nano-vllm：`nanovllm/layers/layernorm.py::RMSNorm`

### Qwen3 Tech Report

（待补 —— T2/T3 涉及 SwiGLU/RoPE 时再展开）

### SwiGLU (Shazeer 2020)

（待补 —— 开 T2 时由 `/plan T2` 自动生成）

### RoPE (Su et al. 2021)

（待补 —— 开 T3 时生成）

### GQA (Ainslie et al. 2023)

（待补 —— 开 T4 时生成）

---

## Libraries

### transformers — Qwen3 模块（ground truth）

**项目锁定版本**：`transformers==5.10.2`（已 lock 在 `pyproject.toml`）；切勿随意升级。

**关键类清单**：

| 类名 | 模块 | 对应 inferlite |
| --- | --- | --- |
| `Qwen3Config` | `configuration_qwen3` | `inferlite/config.py::ModelConfig` |
| `Qwen3RMSNorm` | `modeling_qwen3` | `inferlite/model/layers.py::RMSNorm` |
| `Qwen3MLP` | 同上 | `SwiGLUMLP` |
| `Qwen3RotaryEmbedding` | 同上 | `RotaryEmbedding` |
| `Qwen3Attention` | 同上 | `GQAAttention` |
| `Qwen3DecoderLayer` | 同上 | `Qwen3DecoderLayer` |
| `Qwen3Model` | 同上 | `Qwen3Model` |
| `Qwen3ForCausalLM` | 同上 | 不实现（直接 tie embed） |

**标准对齐测试模式**：

```python
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from inferlite.model.layers import RMSNorm

def test_vs_ref():
    H = 1024
    ref = Qwen3RMSNorm(H, eps=1e-6).eval()
    mine = RMSNorm(H, eps=1e-6).eval()
    mine.load_state_dict(ref.state_dict())   # 同步权重
    x = torch.randn(2, 8, H)
    with torch.no_grad():
        assert torch.allclose(mine(x), ref(x), atol=1e-5)
```

**易错点**：
1. `.eval()` 必调（关 dropout）
2. `.load_state_dict()` 必同步（否则 weight 随机 vs ones）
3. `with torch.no_grad():` 节省内存
4. 创建 ref 时传 `Qwen3Config` 实例，避免字段不一致

**Qwen3-0.6B 关键参数**：
- H=1024, I=3072, N=28, num_heads=16, num_kv_heads=8（GQA 2:1）
- head_dim=128, V=151936, rope_theta=1e6, rms_norm_eps=1e-6
- tie_word_embeddings=True, attention_bias=False

**外部参考**：
- 源码：https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py
- 官方文档：https://huggingface.co/docs/transformers/main/en/model_doc/qwen3

### transformers — 五个核心对象（HF 推理最小闭环）

| 对象 | 作用 | inferlite 是否手写 |
| --- | --- | --- |
| `AutoTokenizer.from_pretrained(id)` | 字符串 → token id（BPE） | 否，永远复用 |
| `AutoModelForCausalLM.from_pretrained(id, dtype, device_map)` | 下载权重 + 实例化 | **是**：`Qwen3Model.load_from_modelscope()` |
| `model.eval()` | 切推理模式 | 是，`__init__` 末尾自动调 |
| `tokenizer.decode(ids, skip_special_tokens=True)` | token id → 字符串 | 否 |
| `model.generate(...)` | 自动循环 forward | **手撕**：CLI 自己写循环+采样 |

`generate()` 的本质（M1 要重写）：
```python
while not done:
    out  = model(input_ids)
    next = sampler(out.logits[:, -1, :])
    input_ids = torch.cat([input_ids, next], dim=1)
```

**from_pretrained 关键参数**：
- repo id：`"Qwen/Qwen3-0.6B"`
- dtype：`torch.float32`（M1-M5 阶段固定 fp32）
- device_map：`"mps" / "cuda" / "cpu"`

**注意**：transformers 5.x 用 `dtype=`，4.x 是 `torch_dtype=`，二者等价。

### pytest — 单测核心 API

**4 个核心特性**（本项目几乎只用这些）：

#### 1. 函数即测试
```python
def test_xxx():
    assert ...
```
- 文件 `test_*.py`，函数 `test_*`
- 运行：`uv run pytest tests/unit/test_rmsnorm.py -v`

#### 2. `@pytest.mark.parametrize`
本项目最常用，N shape × M dtype 笛卡尔积：

```python
@pytest.mark.parametrize("shape", [(2, 8), (4, 16), (1, 32, 64)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_vs_ref(shape, dtype):
    ...
```
- 自动展开 3×3=9 case
- 失败时显示出错参数组

#### 3. `pytest.fixture`
```python
@pytest.fixture
def qwen3_config():
    return Qwen3Config(hidden_size=1024, ...)
```
- scope：`function`（默认）/ `module` / `session`

#### 4. `pytest.mark.slow` + 选择性运行
配合 pyproject.toml 注册 marker；CI 跑 `-m "not slow"` 跳过。

**常用参数**：
- `-v` 详细 / `-x` 第一败停 / `-k <expr>` 名字过滤 / `--lf` 上次失败
- `--tb=short` 简短 traceback

**易错点**：
1. **不用 `torch.equal`**（fp16/bf16 必然不等）→ 用 `torch.allclose(..., atol=...)`
2. **atol 按 dtype 设**：fp32 1e-5，fp16/bf16 5e-3
3. **parametrize 装饰器堆叠**：靠近函数的最先变化（笛卡尔积外层→内层）

**外部参考**：https://docs.pytest.org/

### modelscope — snapshot_download

国内环境下载 Qwen3 等 HF repo：

```python
from modelscope import snapshot_download
local_dir = snapshot_download("Qwen/Qwen3-0.6B")
# 然后正常用 transformers.from_pretrained(local_dir)
```

- Qwen 系列在 ModelScope 的 repo id 与 HF 完全一致
- 权重 sha256 一致
- 第一次下完留 `.lock` 文件，进程 kill 后不释放 → preflight 应加 lock 清理逻辑

### huggingface_hub 1.x — 域名硬校验

**重要陷阱**：1.0 起客户端硬校验 URL 必须 `huggingface.co`。即使设 `HF_ENDPOINT=hf-mirror.com`，部分代码路径（auth / metadata）仍 hard-code 官方域名 → 国内不可用。

**解法**：国内一律走 `modelscope.snapshot_download`，不要依赖 HF_ENDPOINT。

---

## Concepts

### 数值升精度（upcast to fp32）

**一句话**：fp16/bf16 模型中，涉及 reduce + sqrt + softmax / exp / log 的算子，必须临时升精度到 fp32 计算，再降回原 dtype 输出。

**数学/直觉**：
- fp16 尾数 10 位，范围上限 65504：`x²` 若 |x|>256 即溢出
- bf16 尾数 7 位，范围与 fp32 同，但精度差，累加误差严重
- fp32 尾数 23 位，足够

reduce 会**累加**误差。hidden_size=1024 时 fp16 的 mean(x²) 几乎必然显著偏差。

**何时需要 upcast**：

| 场景 | 必要 | 备注 |
| --- | --- | --- |
| RMSNorm / LayerNorm | **必须** | mean(x²) + sqrt |
| Softmax | **必须** | exp 溢出 |
| log_softmax / NLLLoss | **必须** | log 数值敏感 |
| matmul（Linear） | 不必 | tensor core 已优化 |
| element-wise add/mul | 不必 | 不放大误差 |

**标准模式**：
```python
def numerically_sensitive_op(x):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    # ... fp32 运算 ...
    return result.to(input_dtype)   # 最后一步降回
```

**本项目应用**：T1 RMSNorm / T4 Attention softmax / T9 Sampler log_softmax

### tie_word_embeddings

**一句话**：embed 矩阵 `[V, H]` 既当 input embedding 也当 output projection (lm_head)，省一份权重。

**Qwen3-0.6B**：tied = True；>0.6B 是 False。所以不要硬编码。

**实现**：
```python
logits = F.linear(hidden, self.embed_tokens.weight)   # 复用 embed.weight 作为 lm_head
```

不需要单独 `self.lm_head` 层。

### 形状速查（背下来） { #shape-cheatsheet }

| 张量 | 形状 | 含义 |
| --- | --- | --- |
| `input_ids` | `[B, T]` | int64 token ID |
| `inputs_embeds` | `[B, T, H]` | 跳过 embed 直接喂浮点 |
| 隐藏态 | `[B, T, H]` | 层间 |
| logits | `[B, T, V]` | 词表分数 |
| 下一 token | `[B, 1]` | argmax |

**字母**：
- B = batch（M1=1，M3 引入 batching）
- T = seq length
- H = hidden_size（Qwen3-0.6B=1024）
- I = intermediate_size（=3072，仅 SwiGLU 中间）
- V = vocab_size（=151936）
- N = layers（=28）

### 推理上下文管理 { #inference-context }

```python
with torch.no_grad():
    out = model(...)
# 等价 (更彻底):
with torch.inference_mode():
    out = model(...)
```
推理路径必须包，否则 MPS 显存翻倍。

### Factory pattern (经典工厂)

（待补 —— T0' ModelConfig.from_json() 涉及时展开）

---

## Tools

### uv（Python 包管理器）

Astral 出品，Rust 写。取代 pyenv + venv + pip + pip-tools。

- 装 PyTorch + transformers：pip 60s，uv 5s
- 一站式：Python 版本 + venv + 依赖锁定
- lockfile 默认开（`uv.lock` commit 进 git）
- 原生支持 PyTorch 索引源切换（cpu / cu121 / cu124）

**常用命令**：
```bash
uv sync                    # 装依赖
uv sync --frozen           # 严格按 lock
uv run pytest              # 在 venv 跑命令
uv add "torch>=2.5"        # 加依赖
uv add --dev pytest-xdist  # dev 依赖
uv lock --upgrade <pkg>    # 升级单包
```

**`uv.lock` 重要性**：
- `pyproject.toml` 说"我要 torch>=2.4"
- `uv.lock` 锁定"实际装的是 torch==2.7.0 + 全部传递依赖 + sha256"
- 必须 commit（保证跨机器/时间复现）

### make（任务运行器）

Unix 自带（1976）。`Makefile` 定义"任务名 → shell 命令"。

```makefile
setup:
\tbash scripts/setup.sh
test:
\tuv run pytest
```

**关键**：执行行**必须 Tab 缩进**，不能空格。

### ruff（lint + format）

取代 flake8 + black + isort。`make lint` / `make fmt`。配置在 `pyproject.toml [tool.ruff]`。

### pre-commit（commit 前自动检查）

`.pre-commit-config.yaml` 配置，含 trailing-whitespace / EOF / yaml/toml check / large-file guard / ruff lint+format。`scripts/setup.sh` 自动注册 hook。

### pytest-mark / CI matrix

CI 跑 `-m "not slow"` 跳过慢测试。matrix: ubuntu + macos, python 3.12。

---

## 维护规则

- **新增卡片**：在对应章节末尾追加 `### <Title>` 子段，不开新文件
- **删除卡片**：直接删段，更新引用
- **跨段引用**：用 markdown 锚点 `[upcast](#数值升精度upcast-to-fp32)`
- **完整精读论文**：留链接到 `docs/papers/...`（走 `paper-deep-read` skill），本文件只放项目视角摘要
