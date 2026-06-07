---
description: 任务卡完成后双轨沉淀（文件 lessons + Memory）。用法：/archive-task T2
argument-hint: <task-id>，例如 T2
---

针对刚完成的任务卡 `$ARGUMENTS`（如 T2），执行双轨知识沉淀：

## 1. 读源材料
- `inferlite/docs/tasks/M1-$ARGUMENTS-*.md`（任务卡，含"实战教训"或"坑"区块）
- 最近 3 个 commit message（`git log --oneline -3`）
- 测试结果

## 2. 提取教训
从任务卡 + commit + 实测中，识别**非平凡**教训。判定标准：
- ✅ 算法/数值上的坑（如 fp32 upcast）
- ✅ 工程/环境的坑（如 stale lock）
- ✅ 命名/约定决策（如 eps vs variance_eps）
- ✅ 跨任务可复用的方法（如"先 weight 乘再降 dtype"）
- ❌ 一次性、项目特有、太琐碎的（如某个 typo）

**没有非平凡教训也是合理结果**，跳过即可。

## 3. 写入文件（轨道 1）

对每个识别出的教训，在 `~/learning/docs/projects/inferlite/lessons/` 新建一个 markdown：

文件名：`<short-topic>.md`（kebab-case，如 `rmsnorm-fp32-upcast.md`）

模板：
```markdown
# Lesson: <一句话主题>

## 来源
- 任务卡: $ARGUMENTS
- 提交: `<hash>`
- 日期: <YYYY-MM-DD>

## 现象
（观察到的问题）

## 根因
（为什么会这样）

## 解法
（怎么修，含代码片段）

## 适用范围
（这个教训可复用到哪些场景）

## 相关
（链接到 transformers / nano-vllm / 论文 / 本仓库其他文件）

## Memory Tag
`tag1, tag2, tag3`
```

## 4. 写入 Memory（轨道 2）

对每个 lesson 调用：
```
update_memory(
    action="create",
    dimension="repos",
    category="common_pitfalls_experience"  // 或 development_practice_specification
    title="inferlite: <short-topic>",
    content="<lesson 摘要 + 适用范围 + 解法关键代码>",
    keywords="<tag1, tag2, tag3, inferlite, M1>",
    reason="任务卡 $ARGUMENTS 完成后沉淀教训"
)
```

## 5. 更新索引

修改 `~/learning/docs/projects/inferlite/README.md` 的"Lessons"区块：
追加一行 `- [<topic>](./lessons/<topic>.md) — <一句话描述>`

## 6. 更新任务卡 README

修改 `~/learning/inferlite/docs/tasks/README.md`：把 `$ARGUMENTS` 状态改 ✅

## 7. 输出报告

```
任务卡 $ARGUMENTS 归档完成

写入 lessons:
  - lessons/<topic1>.md
  - lessons/<topic2>.md

写入 Memory:
  - <category>: <title>

索引已更新: docs/projects/inferlite/README.md
状态已更新: inferlite/docs/tasks/README.md ($ARGUMENTS → ✅)

下一步建议: /next-task
```

## 8. 不做
- 不要重复沉淀已有 lessons（先 `ls lessons/`，相似主题就追加章节而非新建）
- 不要瞎造教训凑数；非平凡就跳过
- 不要修改 inferlite/ 业务代码
