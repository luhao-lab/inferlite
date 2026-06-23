# Lessons Learned

> 踩过的坑 + 解法。叙事性，有现场感。区别于 knowledge.md（事实性、可独立读）。
> 单文件多 H2，按时间顺序追加，永久保留。

---

## 📊 索引

| ID | 来源 | 一句话 | 适用范围 |
| --- | --- | --- | --- |
| **L1** | T1 RMSNorm | RMSNorm 必须 upcast fp32 算方差 | 所有 Norm 类算子 |
| **L2** | M0 环境搭建 | 国内拉模型用 ModelScope 替代 HF mirror | 任何拉权重的脚本 |
| **L3** | M0 知识库重构 | 地基 vs 算法是两个频道，不要混着切 | 协作节奏 / AI 协作 |
| **L4** | T0 ModelConfig | GQA 的 `head_dim` 是独立超参，不能从 KV 头数推导 | GQA/MQA Attention / Config 设计 |

---

## L1: RMSNorm 必须 upcast fp32 算方差

**来源**：T1 RMSNorm（2026-06-07，commit `259def0`/`bd487d1`）

### 现象
fp16/bf16 输入时若 `mean(x²)` 在原 dtype 上算：
- fp16 范围上限 65504，hidden_size=1024 时 `x²` 累加易溢出
- bf16 范围够但尾数仅 7 位，平方损失精度严重
- 28 层 RMSNorm 累计后 logits 偏差 > atol=1e-3

### 根因
RMS = sqrt(E[x²])。x² 在 fp16 上几乎必然损失精度；reduce(mean) 进一步累积误差。

### 解法
RMSNorm 内部一律 upcast fp32 做 reduce，**最后**再 cast 回原 dtype：

```python
def forward(self, x):
    input_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + self.eps)
    return (self.weight * x).to(input_dtype)   # 最外层降回
```

注意：`self.weight * x` 在 fp32 算，最外层 `.to(input_dtype)`；不能"先 cast x 再乘 weight"。

### 适用范围
所有 reduce + sqrt 的归一化层（RMSNorm / LayerNorm / GroupNorm），以及 softmax / log_softmax。

### 相关
- knowledge.md → Concepts → 数值升精度
- knowledge.md → Papers → RMSNorm

---

## L2: 国内用 ModelScope 替代 HF mirror（hub 1.x 硬校验）

**来源**：M0 preflight 调试（2026-06-06~07，commit `8814cf5`/`5b7fc5e`）

### 现象
国内 `make preflight` 报 `Connection reset by peer`：
1. 走默认 `huggingface.co` — 被墙
2. 设 `HF_ENDPOINT=https://hf-mirror.com` — 仍失败
3. 直接 `curl https://hf-mirror.com/...` — 能下
4. `hf download Qwen/...` — 失败

### 根因
`huggingface_hub>=1.0` 在客户端硬校验 repo URL 必须匹配 `huggingface.co`。即使设 `HF_ENDPOINT`，部分代码路径（auth / metadata）仍 hard-code 官方域名，绕不过去。

### 解法
切到 **ModelScope**（独立 SDK，无 HF 域名依赖）：

```python
from modelscope import snapshot_download
local_dir = snapshot_download("Qwen/Qwen3-0.6B")
# 然后 transformers.from_pretrained(local_dir, local_files_only=True)
```

附带：第一次下载完成后 ModelScope 留 lock 文件；进程被 kill 时 lock 不释放，下次卡死。`scripts/preflight.py` 加 lock 自动清理逻辑。

### 适用范围
- 所有"国内开发 + HF 上有镜像"的场景
- `huggingface_hub>=1.x` 全部受影响（旧版 0.x 还能用 HF_ENDPOINT）

### 相关
- knowledge.md → Libraries → modelscope / huggingface_hub
- inferlite/scripts/preflight.py::`ensure_local_model()`

---

## L3: 地基与算法是两个频道，不要混着切

**来源**：T1 RMSNorm 复盘（2026-06-07，commit `cdebc79`）

### 现象
T1 一张卡，从开工到 12/12 绿，期间穿插 7 个独立基础设施话题：
1. HF mirror Connection reset
2. ModelScope 切换
3. `accelerate` 缺失
4. ModelScope stale lock
5. conda base pytest 冲突
6. inferlite/ 包目录未建
7. CI / pre-commit 配置

算法本身（4 行 RMSNorm）只占 5% 时间，95% 在地基切换。

### 根因
**协作前没把"地基"修完就开"算法"**：
- 包骨架（`inferlite/` 空目录）应在 M0 一次性建好
- preflight 应作为 T1 的硬前置
- CI / pre-commit 应作为 M0 一部分
- python 环境（uv vs conda）应在 setup.sh 阶段就锁定

地基与算法两个频道反复切换，认知成本远高于一次性修完。

### 解法

**原则**：每个里程碑开始前花 10 分钟做地基（目录、CI、preflight、环境），然后**纯粹**写算法。

**已落地**：
1. `scripts/setup.sh` 加包骨架建立（`mkdir` + 幂等 `__init__.py`）
2. `scripts/setup.sh` 自动注册 pre-commit hook
3. 每张任务卡有"启动 checklist"（包括"上一卡测试绿 / preflight 通"）
4. `/preflight` slash 命令，开工前一键确认地基

**任务卡升级**：7 字段的"前置"必须列**所有地基依赖**，不只是任务依赖：包骨架 / preflight / 上一卡测试 / 相关 knowledge。

### 适用范围
任何"AI + 人"协作的学习型项目；任何 Vibe coding → spec-driven 的转折点。

### 相关
- decisions.md → ADR-001 spec-driven 工作流
- CLAUDE.md 反模式 #4

---

## L4: GQA 的 `head_dim` 是独立超参，不能从 KV 头数推导

**来源**：T0 ModelConfig（2026-06-09）

### 现象
开 T0 时，最初容易写出两个错误直觉：

1. `head_dim == hidden_size / num_attention_heads` 是永远成立的 invariant
2. GQA 里如果要兜底 `head_dim`，是不是应该除 `num_key_value_heads`

Qwen3-0.6B 直接反例：

```text
hidden_size = 1024
num_attention_heads = 16
num_key_value_heads = 8
head_dim = 128

1024 / 16 = 64  != 128
1024 / 8  = 128 == 128   # 只是这个模型数值上碰巧等于，不是定义来源
```

### 根因
在 GQA/MQA 里：

```text
Q shape: [B, T, n_q,  d]
K shape: [B, T, n_kv, d]
V shape: [B, T, n_kv, d]
```

`num_key_value_heads` 减少的是 **KV head 的组数**，不是每个 head 的维度。每个 Q/K/V head 的最后一维仍然是同一个 `head_dim`。

因此：
- 真实模型如果在 config.json 里给了 `head_dim`，必须优先读 JSON
- 缺字段时的兼容兜底采用传统定义 `hidden_size // num_attention_heads`
- 不要写 `head_dim == hidden_size // num_attention_heads` 作为 Qwen3 的 invariant
- 更不要用 `hidden_size // num_key_value_heads` 当通用推导公式

### 解法
T0 `ModelConfig.from_json()` 的策略：

```python
if "head_dim" not in raw:
    raw["head_dim"] = raw["hidden_size"] // raw["num_attention_heads"]
```

并在代码注释中明确：这只是老 config 缺字段时的兼容兜底；Qwen3-0.6B config.json 明确有 `head_dim=128`，不会走这个分支。

### 适用范围
- 所有 GQA/MQA attention 实现
- 所有从 HF `config.json` 反序列化模型超参的代码
- T4 `GQAAttention` 的 q/k/v projection shape 设计

### 相关
- knowledge.md → Papers → Qwen3 Tech Report → Self-Attention 内部
- docs/tasks/M1-T0-ModelConfig.md

---

## 维护规则
- **新教训追加**：在文件末尾 `## L<N>: <title>`，编号递增
- **格式固定 4 段**：现象 / 根因 / 解法 / 适用范围 + "相关"
- **被否决的教训**：保留，加 `[已修正]` 前缀和说明
- 太琐碎/一次性的不记
