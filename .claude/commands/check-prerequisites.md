---
description: 任务卡开工前检查前置知识是否到位。用法：/check-prerequisites T2
argument-hint: <task-id>，例如 T2
---

针对即将开工的任务卡 `$ARGUMENTS`（如 T2），确认前置知识齐全；不齐则自动补卡。

## 1. 读任务卡

```
read_file ~/learning/inferlite/docs/tasks/M*-$ARGUMENTS-*.md
```

从任务卡提取：
- 算法核心区块涉及的论文 / 概念
- 产出文件涉及的库（一定包含 `transformers.<XXX>` 作为 ground truth）
- "前置"区块列出的依赖
- "坑"区块涉及的概念

## 2. 列必备 knowledge 卡清单

至少应包含：
- 算法对应论文/概念卡
- transformers ground truth 类卡（`knowledge/libs/transformers-qwen3.md` 通用）
- `knowledge/libs/pytest-core.md`（所有任务卡都要写测试）
- 任务卡"坑"中涉及的概念卡

## 3. 比对现有

```bash
ls ~/learning/docs/projects/inferlite/knowledge/{papers,libs,concepts,tools}/
```

分类：
- ✅ 已有：直接列链接
- 🆕 缺失：进入补卡流程

## 4. 自动补卡（缺失项）

对每张缺失的卡，调用 `/research-before-plan` 的"3. 新建知识点卡"流程：
- 项目内 grep
- web 调研
- 写卡（按 `_TEMPLATE.md`）
- update_memory

## 5. 输出"开工阅读清单"

```
$ARGUMENTS 前置知识检查

必读 (按推荐阅读顺序):
  1. knowledge/papers/<X>.md  — <一句话>
  2. knowledge/libs/<Y>.md    — <一句话>
  3. knowledge/concepts/<Z>.md — <一句话>

推荐参考实现:
  - nano-vllm <file:line>
  - transformers <file:line>

预判风险（来自历史 lessons）:
  - lessons/<topic1>.md
  - lessons/<topic2>.md

补卡操作: N 张新建 / M 张已有

→ 现在可以开工：/next-task 展开 $ARGUMENTS
```

## 6. 不做
- 不要凭印象判断"前置知识到位"——必须实际 `ls` 知识库
- 不要补本任务卡之外的卡（避免范围扩张）
- 不要修改 inferlite/ 业务代码
