---
description: 里程碑完成后生成 milestone summary。用法：/archive-milestone M1
argument-hint: <milestone-id>，例如 M1 / M1a / M2
---

针对刚完成的里程碑 `$ARGUMENTS`（如 M1a），生成聚合总结并入库。

## 1. 读源材料

- `inferlite/docs/$ARGUMENTS.md`（里程碑作战地图）
- `inferlite/docs/tasks/$ARGUMENTS-T*.md`（本 M 所有任务卡，全部应为 ✅）
- `~/learning/docs/projects/inferlite/lessons/*.md` 中本 M 期间新建的
- `~/learning/docs/projects/inferlite/decisions/*.md` 中本 M 期间新建的
- `inferlite/docs/PROGRESS.md` 本 M 日志
- `git log --oneline <M start>..HEAD` 本 M 提交历史

## 2. 验收检查

- 所有任务卡状态都是 ✅？（若否，拒绝归档）
- DoD 全部满足？（参考 `inferlite/docs/$ARGUMENTS.md` §1）
- CI 最新 run 绿？

任一不满足 → 输出未完成项，停止归档。

## 3. 生成 milestone summary

写入 `~/learning/docs/projects/inferlite/milestones/$ARGUMENTS-summary.md`：

```markdown
# $ARGUMENTS Summary

## 元信息
- 起止: <YYYY-MM-DD> ~ <YYYY-MM-DD>
- 提交数: N
- 代码增: +X / -Y 行
- 测试增: M cases
- Tag: vX.Y

## 目标与达成
（引用 $ARGUMENTS.md §1 DoD，逐条标 ✅ + 简要数据）

## 架构产出
（本 M 完成了哪些模块/接口/类，引用具体文件:line）

## 关键决策（ADR）
- ADR-NNN ...
- ADR-NNN ...

## 关键教训（lessons）
- lessons/<topic1>.md — 一句话
- lessons/<topic2>.md — 一句话

## 数据/性能（如有）
（M5+ 才有，benchmarks/ 数据）

## 与计划的偏差
- 计划 vs 实际耗时
- 多做了什么 / 少做了什么 / 为什么

## 下一里程碑准备
- 已就绪的前置
- 待补的前置
- 推荐的参考资料（指向 REFERENCES.md 对应阶段）
```

## 4. 写入 Memory

```
update_memory(
    action="create",
    dimension="repos",
    category="project_introduction",
    title="inferlite $ARGUMENTS done",
    content="<summary 顶部 5 行 + 关键决策列表 + 主要教训>",
    keywords="inferlite, $ARGUMENTS, milestone, <主题词>",
    reason="里程碑 $ARGUMENTS 归档"
)
```

## 5. 更新索引

`~/learning/docs/projects/inferlite/README.md`:
- "当前状态" 表格更新 $ARGUMENTS → ✅
- "Milestones" 区块追加 `- [$ARGUMENTS](./milestones/$ARGUMENTS-summary.md)`

`inferlite/docs/PROGRESS.md`:
- 里程碑总表对应行 → ✅ + tag + 日期

## 6. 提交

```
git add -A
git commit -m "docs($ARGUMENTS): milestone done; archive summary

- $ARGUMENTS-summary.md: <gist>
- N lessons sealed
- M decisions archived
- Memory written: project_introduction.inferlite $ARGUMENTS done"
git tag -a vX.Y -m "$ARGUMENTS done"
git push origin main --tags
```

## 7. 输出报告 + 下一步建议

```
$ARGUMENTS 归档完成 ✅

Summary: docs/projects/inferlite/milestones/$ARGUMENTS-summary.md
Tag: vX.Y
Lessons sealed: N
Decisions: M
Commits: K

下一步建议: /plan-next-milestone <next-M>
```
