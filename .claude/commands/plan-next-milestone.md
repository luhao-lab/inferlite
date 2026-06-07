---
description: 基于已完成里程碑的知识库，规划下一个里程碑。用法：/plan-next-milestone M2
argument-hint: <milestone-id>，例如 M2
---

针对将要开始的里程碑 `$ARGUMENTS`（如 M2），**带着前期积累**生成完整 plan。

## 1. 收集上下文

### 1.1 上一里程碑 summary
```
read_file ~/learning/docs/projects/inferlite/milestones/<prev-M>-summary.md
```

### 1.2 所有相关 lessons
```
list_files ~/learning/docs/projects/inferlite/lessons/
read_file 每个看起来与 $ARGUMENTS 主题相关的 lesson
```

判断相关性参考关键词：
- M2 KV cache → lessons/ 中带 `cache, kv, attention, memory` 的
- M3 Continuous Batching → 带 `scheduler, batching, queue` 的
- M4 PagedAttention → 带 `paged, attention, kernel` 的

### 1.3 决策约束
```
read_file ~/learning/docs/projects/inferlite/decisions/*.md
```
关注 ADR 是否对本 M 的实现路径有约束。

### 1.4 Memory 隐式经验
```
search_memory(
    query="inferlite $ARGUMENTS <主题词>",
    keywords="inferlite, $ARGUMENTS, <主题>",
    depth="deep"
)
```

### 1.5 大盘与参考
```
read_file ~/learning/inferlite/docs/PLAN.md  # 找到 $ARGUMENTS 章节
read_file ~/learning/inferlite/docs/REFERENCES.md  # 找到 $ARGUMENTS 阶段的推荐资料
```

## 2. 草拟 plan

按 M1.md 的结构（11 章），生成 `inferlite/docs/$ARGUMENTS.md`：
1. DoD（从 PLAN.md $ARGUMENTS 章节抄/扩展）
2. 架构总览（数据流图，标注每个模块属于哪张任务卡）
3. 模块清单 + 任务 DAG（Mermaid）
4. 任务卡总表（索引，指向 docs/tasks/$ARGUMENTS-T*.md）
5. 关键策略（如本 M 有"模型加载策略"这种核心子话题）
6. 单卡详细模板（指向 docs/tasks/）
7. 测试金字塔
8. 易踩坑（**从 lessons/ 摘录相关条目**，标"来自 lesson XXX"）
9. 推进节奏
10. 启动 checklist
11. 概念速查（指向 CONCEPTS.md）
12. 参考资料（指向 REFERENCES.md）

## 3. 拆任务卡

为本 M 每张任务卡新建 `inferlite/docs/tasks/$ARGUMENTS-T<N>-<name>.md`（前 2-3 张详细展开，其余等开工时再补）：
- 用 `docs/tasks/_TEMPLATE.md` 模板
- "坑"区块**从 lessons/ 引用**，不要重新发明
- "前置"列出地基依赖 + 任务依赖

## 4. 更新索引

`inferlite/docs/tasks/README.md`:
- 追加 $ARGUMENTS 区块及所有 T*.md 链接

`~/learning/docs/projects/inferlite/README.md`:
- "当前状态" 表格新增 $ARGUMENTS → ⬜
- "上一里程碑准备" 区块（前 M summary 末尾）有要求的，确认本 M 都满足

## 5. 输出 plan 摘要给用户

```
$ARGUMENTS Plan 已生成（基于前期知识库）

引用的 lessons:
  - lessons/<topic1>.md
  - lessons/<topic2>.md

引用的 decisions:
  - ADR-NNN

Memory 命中:
  - <category>: <title>

新建文件:
  - inferlite/docs/$ARGUMENTS.md (XX 行)
  - inferlite/docs/tasks/$ARGUMENTS-T1-*.md
  - inferlite/docs/tasks/$ARGUMENTS-T2-*.md

下一步建议:
  - 审阅 $ARGUMENTS.md §2 架构图
  - /preflight-check 确认地基
  - /next-task 开第一张
```

## 6. 不做
- 不要自己写 `inferlite/` 业务代码
- 不要跳过 1.x 收集步骤直接生成 plan（这是"基于知识库"的核心）
- 不要凭空生成"坑"列表，必须从 lessons/ 引用或 search_memory 取
