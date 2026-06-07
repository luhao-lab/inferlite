---
description: 规划新里程碑/任务卡前的自动调研。用法：/research-before-plan M2 或 /research-before-plan T2
argument-hint: <scope>，例如 M2 / T2
---

针对即将规划的 `$ARGUMENTS`，做前置调研，产出 `research/<scope>-brief.md` + 补全 `knowledge/` 卡。

## 0. 判定 scope 类型
- `M[0-9]+[a-z]?` → 里程碑级
- `T[0-9]+'?` → 任务卡级

## 1. 列出涉及主题

读以下文件提取主题清单：
- `~/learning/inferlite/docs/PLAN.md` 中 $ARGUMENTS 章节（里程碑级）
- `~/learning/inferlite/docs/<M>.md` §6 任务卡章节（任务卡级）
- `~/learning/inferlite/docs/REFERENCES.md` 对应阶段

主题分 4 类，每类列出来：
- **paper**: 涉及哪些论文 / 技术报告
- **lib**: 用到哪些库的 API（transformers / pytest / modelscope / huggingface_hub / pytorch 等）
- **concept**: 哪些算法 / 工程概念（GQA / RoPE / KV cache / tie_embedding / factory pattern 等）
- **tool**: 哪些工具链（uv / ruff / pre-commit / pytest-benchmark 等）

## 2. 对每个主题：知识库已有吗？

```bash
ls ~/learning/docs/projects/inferlite/knowledge/{papers,libs,concepts,tools}/
```

- ✅ 已有 → 标记并直接引用
- 🟡 部分有 → 标记需补充章节
- 🆕 新建 → 进入第 3 步

## 3. 新建知识点卡

对每个 🆕 主题：

### 3.1 调研
**优先级**:
1. 先查项目内：`grep -r "<topic>" ~/learning/inferlite/`、本地 transformers 源码
2. 再 web 调研：`search_web(query="<topic>")` → `fetch_web(url=top hit)`
3. 必要时翻原始论文（arxiv abs 页足够）

### 3.2 写卡
按 `~/learning/docs/projects/inferlite/knowledge/_TEMPLATE.md` 写入对应子目录：
- papers/ ← 论文
- libs/ ← 库 API
- concepts/ ← 算法/工程概念
- tools/ ← 工具

**写卡纪律**:
- 一张卡 100-300 行，不要抄完整官方文档
- 必须含"在本项目用在"区块（指向具体 file:line 或任务卡）
- 必须含"外部参考"区块（URL）
- 必须含 Memory Tag

### 3.3 写入 Memory（双轨）
对每张新建的 knowledge 卡：
```
update_memory(
    action="create",
    dimension="repos",
    category="project_introduction"  // 或更精确的类别
    title="inferlite knowledge: <topic>",
    content="<卡片的"一句话"+ 关键 API/公式 + 在本项目位置>",
    keywords="<Memory Tag>",
    reason="$ARGUMENTS 规划前调研"
)
```

## 4. 输出 research brief

按 `~/learning/docs/projects/inferlite/research/_TEMPLATE.md` 写入：
`~/learning/docs/projects/inferlite/research/$ARGUMENTS-brief.md`

包含：
- 主题清单表（4 类 × 状态）
- 关键发现（5 条以内，对 plan 设计有影响的事实）
- 推荐参考实现（项目 + 文件 + 看什么）
- 风险预判（任务卡的"坑"会引用这里）
- 范围与边界

## 5. 输出报告

```
$ARGUMENTS 调研完成

主题清单: P 篇论文 + L 个库 + C 个概念 + T 个工具
  ✅ 已有: N
  🟡 部分有: M
  🆕 新建: K

新建 knowledge 卡:
  - knowledge/papers/<topic1>.md
  - knowledge/libs/<topic2>.md
  ...

Memory 写入: K 条

简报: docs/projects/inferlite/research/$ARGUMENTS-brief.md

关键发现 (top 3):
  1. ...
  2. ...
  3. ...

下一步建议: /plan-next-milestone $ARGUMENTS (或 /next-task)
```

## 6. 不做
- 不要把 knowledge 卡写成论文/官方文档全文复印
- 不要遗漏 web 调研（如果项目内信息不足）
- 不要跳过 update_memory（这是双轨的第 2 轨）
- 不要在调研阶段动 inferlite/ 业务代码
