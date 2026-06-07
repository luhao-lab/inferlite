---
description: 规划下一阶段（M 级或 T 级），含前置调研
argument-hint: "<scope>  例: M2 / T0' / 整体重排"
---

# /plan — 规划（M / T / 调整）

入参 `$ARGUMENTS` 指明规划范围。

## Step 1: 解析 scope

- `M<n>` → 规划整个里程碑（如 `M2`）
- `T<x>` / `M<n>-T<x>` → 规划单张任务卡
- 其他自由文本 → 当 topic 处理（如"整体重排 M3-M5"）

## Step 2: 前置调研（HARD 前置，不可跳过）

无论 scope 大小，先做调研：

### 2.1 拆 topic
按 4 类列出需要的知识：
- **papers**：相关论文
- **libs**：用到的库 API
- **concepts**：数学/工程概念
- **tools**：工具链

### 2.2 检查 docs/3-kb/knowledge.md 已有章节
对照各 H2 的 `### <title>` 子段。

### 2.3 缺口 → web 调研
对每个缺失 topic：
1. `search_web` 找官方源 / 论文 / 权威博客
2. `fetch_web` 抓取关键内容
3. 在 `docs/3-kb/knowledge.md` 对应 H2 末尾追加 `### <title>` 章节，按 knowledge 卡模板写：
   - 一句话
   - 关键论点 / 公式 / 接口
   - 在本项目用在 → 列出未来要修改的 inferlite/ 路径
   - 外部参考链接
4. 关键 knowledge → `update_memory`

### 2.4 输出研究简报
直接在响应里给出简报（不再单独建文件），包含：
- 调研 topic 列表（4 类 × 状态：已有 / 新增 / 跳过）
- 关键发现 ≤ 5 条
- 推荐实现路径
- 风险预判
- 边界（不做什么）

## Step 3: 基于研究做规划

### 3.1 若 scope = M<n>
1. 读 `docs/1-plan/PLAN.md` 找到 M<n> 段
2. 读上一 M 的 `docs/M<m>.md` Summary 段（如有）
3. 生成 `docs/M<n>.md`，含：
   - §1 目标（一句话 / 完成条件 / 演示场景）
   - §2 关键技术（引用 knowledge.md 章节）
   - §3 模块拆分（架构图 + 接口签名）
   - §4 任务卡总表（T1, T2, ... 状态/工时/前置）
   - §5 验证策略（L0-L3 各做什么）
   - §6 易踩坑（引用 lessons.md L<N>）
   - §7 路径图（mermaid sequenceDiagram，可在 M 完成后回填）
4. 给每张任务卡建 `docs/2-tasks/M<n>-T<x>-<title>.md` 骨架（用 `_TEMPLATE.md`）

### 3.2 若 scope = T<x>
1. 读 `docs/M<current>.md` §6 找该任务的简短描述
2. 生成 / 补全 `docs/2-tasks/M<n>-T<x>-<title>.md`：
   - 用 `docs/2-tasks/_TEMPLATE.md` 7 字段
   - 前置：列 knowledge.md / lessons.md 引用
   - 接口签名：从研究简报推导
   - 测试清单：vs `Qwen3<X>` 的对齐 case

### 3.3 若 scope = 调整
- 列影响的文件清单（PLAN.md / M*.md / tasks/*）
- 写改动 diff，问用户确认

## Step 4: 等用户确认

输出后**停**。

**禁止**：跳过调研直接规划；不引用 knowledge.md 凭空写技术细节；不更新 docs/1-plan/PROGRESS.md。
