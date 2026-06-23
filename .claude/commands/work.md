---
description: 开始一张任务卡（含前置 knowledge gap 检查）
argument-hint: "<task-id>  例: T0' 或 M1-T0p-ModelConfig"
---

# /work — 开始任务

入参 `$ARGUMENTS` = 任务卡 ID（如 `T0'` / `M1-T0p` / `M1-T2`）。

## Step 1: 定位任务卡

1. 在 `docs/tasks/` 找 `M<n>-<id>-*.md` 文件
2. 若找不到：列出 `docs/M<current>.md` §6 任务清单，请用户确认 id 与文件名
3. 找到后用 `read_file` 读完整任务卡

## Step 2: 前置 knowledge 检查（防止盲开工）

按任务卡"前置 / 依赖"段，列出需要的 knowledge 卡：
- **必查清单**：相关 paper / 库 API / 概念 / 工具
- 对照 `docs/kb/knowledge.md` 各 H2 是否已有对应章节

**如有缺口** → 输出：
```
缺失 knowledge:
- [ ] <title>  (paper/lib/concept/tool)
- [ ] <title>
```
告诉用户："建议先补 knowledge：是否需要我做 web 调研（按 `/plan` 的研究流程）？(Y/n)"

用户同意后：
- web_search + fetch_web 调研
- 把新章节追加到 `docs/kb/knowledge.md` 对应 H2
- 关键 knowledge 同步 `update_memory`（category=`project_introduction`，keywords 含 `inferlite`）

**无缺口** → 直接进 Step 3。

## Step 3: 输出"作战简报"

模板：
```markdown
## 任务: <id> <title>

**目标（一句话）**：...

**前置阅读清单**：
- knowledge.md → Papers → <X>
- knowledge.md → Libraries → <Y>
- lessons.md → L<N>（如有相关）

**接口签名**：
```python
class Foo:
    def bar(self, ...) -> ...:
        """<docstring>"""
```

**测试入口**：`tests/unit/test_<x>.py`（必须 vs transformers ground truth）

**完成条件（DoD）**：
- [ ] 实现 `inferlite/.../<x>.py`
- [ ] L0 测试 N/N 通过
- [ ] 类型检查通过
- [ ] 无 lint 错误

**预估工时**：X 分钟

**易踩坑**（来自 lessons.md / 任务卡）：
- ...
```

## Step 4: 等用户确认后开工

输出上述简报后**停**。等用户 "开始" 或 "改一下 X" 再动。

**禁止**：未读任务卡就猜测；未列前置 knowledge 就开工；自己写实现代码（违反 ADR-001 "AI 不写业务代码"）。
