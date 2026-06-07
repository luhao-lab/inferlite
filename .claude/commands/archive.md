---
description: 归档任务卡 / 里程碑（含 lessons + knowledge 沉淀 + mainline summary）
argument-hint: "task <id>  或  milestone M<n>"
---

# /archive — 归档

入参 `$ARGUMENTS`：
- `task T0'` / `task M1-T2` — 归档单张任务卡
- `milestone M1` — 归档整个里程碑

## Mode A: 归档任务卡

### A.1 Preflight
- `make test` 全绿
- `make lint` 干净
- 本任务相关 commit 已推

### A.2 任务卡内容补全
读 `docs/2-tasks/M<n>-T<x>-*.md`，在文件末尾追加：
```markdown
---

## ✅ 完成总结

**状态**: ✅ 完成 (yyyy-mm-dd)
**Commit**: `<sha>`
**实际工时**: X 分钟 / 估计 Y 分钟

### 代码流位置
（这部分在主线上扮演什么角色，一句话）

### 关键决策
- ...

### 用到的知识点
- knowledge.md → Papers → <X>
- knowledge.md → Libraries → <Y>

### 测试结果
- L0: N/N 通过
- 边界 case: ...

### 新增 lessons
- 见 lessons.md L<N>（如有）
```

### A.3 沉淀 lessons（如有新坑）
追加 `docs/3-kb/lessons.md`：
- 新教训 `## L<N+1>: <title>`，4 段格式（现象 / 根因 / 解法 / 适用范围）
- 关键教训 → `update_memory`（category=`common_pitfalls_experience`，keywords 含 `inferlite`）

### A.4 沉淀 knowledge（如本卡新引入未记录的 API/概念）
- `git diff` 看本卡新 import 的库/类
- 若 `docs/3-kb/knowledge.md` 没对应章节，追加 `### <title>` 子段
- 更新 `update_memory`（category=`project_introduction`）

### A.5 更新状态文件
- `docs/1-plan/PROGRESS.md` 勾选本任务
- `docs/M<n>.md` §6 任务清单状态 ✅
- `docs/2-tasks/README.md` 表格状态

### A.6 commit + push
```
docs: archive T<x> <title> + lessons L<N> + knowledge <topic>
```

## Mode B: 归档里程碑

### B.1 Preflight
- 本 M 所有任务卡均 ✅
- L1-L3 验证已跑（见 M<n>.md §5）
- M 级 demo 已录（如有）

### B.2 在 M<n>.md 末尾追加 Summary
```markdown
---

## ✅ Milestone Summary

**完成日期**: yyyy-mm-dd
**总工时**: X / 估计 Y

### 一句话回顾
（这个 M 在整个项目中扮演什么角色）

### 代码流（mermaid sequenceDiagram）
（直接读 tasks/M<n>-T*.md 各"代码流位置"段聚合，画完整路径图）

\`\`\`mermaid
sequenceDiagram
    User->>CLI: ...
    CLI->>...
\`\`\`

### 关键决策
- 列出本 M 期间产生的 ADR（链接到 decisions.md）
- 否决的方案 + 否决理由

### 新增 knowledge（本 M 期间）
- knowledge.md → Papers → <X1>, <X2>
- knowledge.md → Libraries → <Y1>

### 新增 lessons（本 M 期间）
- lessons.md → L<N1>, L<N2>

### 数据
- L0 测试：M 个文件 × N case = 全绿
- L1 集成：N 通过
- L3 端到端：✅ 输出对齐 transformers

### 下一 M 准备
- 已建 M<n+1>.md：[ ] 是 / [ ] 否
- 已跑 `/plan M<n+1>`：[ ] 是 / [ ] 否
```

### B.3 更新元数据
- `docs/1-plan/PROGRESS.md` 勾选整个 M
- `docs/1-plan/PLAN.md` M<n> 行加 ✅ 状态
- 打 git tag `m<n>-done`
- `update_memory`（category=`project_introduction`）记 M-level 关键产出

### B.4 commit + push
```
docs: archive M<n> + summary + tag
```

---

**禁止**：
- 任务卡未 ✅ 全绿就归档
- 不更新 PROGRESS.md
- 不沉淀就跳到下一卡（违反 ADR-001 双轨原则）
