---
description: 任务卡完成后双轨沉淀（文件 lessons + Memory）。用法：/archive-task T2
argument-hint: <task-id>，例如 T2
---

针对刚完成的任务卡 `$ARGUMENTS`（如 T2），执行**三轨**知识沉淀（lessons + knowledge + mainline）：

## 1. 读源材料
- `inferlite/docs/tasks/M1-$ARGUMENTS-*.md`（任务卡，含"实战教训"或"坑"区块）
- 最近 3 个 commit message（`git log --oneline -3`）
- 测试结果
- 本卡实际改动的代码（`git diff <prev>..HEAD -- inferlite/`）

## 2. 提取教训 → 写 lessons/

从任务卡 + commit + 实测中，识别**非平凡**教训。判定标准：
- ✅ 算法/数值上的坑（如 fp32 upcast）
- ✅ 工程/环境的坑（如 stale lock）
- ✅ 命名/约定决策（如 eps vs variance_eps）
- ✅ 跨任务可复用的方法（如"先 weight 乘再降 dtype"）
- ❌ 一次性、项目特有、太琐碎的（如某个 typo）

**没有非平凡教训也是合理结果**，跳过即可。

写入 `~/learning/docs/projects/inferlite/lessons/<topic>.md`（模板见现有 lessons）。

## 3. 检查 knowledge 卡漏网

对本卡实际用到的 API / 概念 / 库：
- 遍历 `git diff` 中的 import 和函数调用
- 列出涉及的：transformers 类、torch 函数、pytest 装饰器、modelscope 函数等
- 对每个：检查 `~/learning/docs/projects/inferlite/knowledge/` 是否有对应卡
- 缺的 → 现场补卡（按 `knowledge/_TEMPLATE.md`）

## 4. 追加 mainline 草稿

在 `~/learning/docs/projects/inferlite/mainline/<Mn>-*.md` 草稿区追加：

```markdown
### $ARGUMENTS <短名> ✅ (<日期>, commit <hash>)
- **代码流位置**: <本模块在整体流程中的位置>
- **关键决策**: <一句话，引用 lessons/ 或 decisions/>
- **用到的知识点**:
  - knowledge/papers/<x>.md
  - knowledge/libs/<y>.md
- **测试**: N/N cases (描述)
- **代码片段**:
  \`\`\`python
  <核心 5-10 行>
  \`\`\`
```

## 5. 写入 Memory（双轨第 2 轨）

对每个 lesson + 每张新建的 knowledge 卡，分别 update_memory：
```
update_memory(
    action="create",
    dimension="repos",
    category="common_pitfalls_experience" (lesson) or "project_introduction" (knowledge),
    title="inferlite: <topic>",
    content="<摘要>",
    keywords="<tag>",
    reason="$ARGUMENTS 完成后沉淀"
)
```

## 6. 更新索引

`~/learning/docs/projects/inferlite/README.md` 的"Lessons"区块追加新条目。

## 7. 更新任务卡 README

修改 `~/learning/inferlite/docs/tasks/README.md`：把 `$ARGUMENTS` 状态改 ✅

## 8. 输出报告

```
任务卡 $ARGUMENTS 归档完成 (三轨沉淀)

[Lessons 轨]
  - lessons/<topic1>.md (新)
  ...

[Knowledge 轨]
  补卡: knowledge/<sub>/<topic>.md (N 张新)
  已覆盖: M 张

[Mainline 轨]
  已追加 mainline/<Mn>-*.md 草稿区

[Memory 轨]
  写入: N + M 条

索引已更新: docs/projects/inferlite/README.md
状态已更新: inferlite/docs/tasks/README.md ($ARGUMENTS → ✅)

下一步建议: /next-task
```

## 9. 不做
- 不要重复沉淀已有 lessons / knowledge（先 ls 检查，相似主题就追加章节而非新建）
- 不要瞎造教训凑数；非平凡就跳过
- 不要修改 inferlite/ 业务代码
