# Architecture Decisions (ADR)

> 长效架构/方法论决策。一个 H2 = 一条 ADR。
> 不轻易增删，重大决策才记。

---

## 📊 索引

| ID | 状态 | 决策 | 时间 |
| --- | --- | --- | --- |
| **ADR-001** | Accepted | spec-driven 工作流 + 双文件知识库 | 2026-06-07 |
| **ADR-002** | Accepted | 知识库与代码同仓（R1：35 → 13 文件） | 2026-06-07 |
| **ADR-003** | Accepted | 分组目录 + MkDocs Material 可视化 | 2026-06-07 |

---

## ADR-001: 采用 spec-driven 工作流 + 双文件知识库

**状态**：Accepted (2026-06-07)

### 背景
inferlite = 作者手撕 + AI 辅助 plan/review/doc 的学习项目。T1 RMSNorm 完成后复盘发现：
1. 任务卡靠会话串行抛出（T1/T2/... 只在聊天，docs 未沉淀）
2. 没有自顶向下的架构 → 任务拆解 → 任务卡的清晰链路
3. 经验教训散落在 commit message 和聊天历史里，新 M 开工无法复用
4. AI 上下文有限，跨会话/跨里程碑知识没有载体

### 决策

#### 文档层（spec-kit 风格）
- `docs/M{n}.md` 作战地图（架构 / 总览 / 任务卡总表）
- `docs/tasks/M{n}-T*.md` 任务卡（一卡一文件，PR 粒度）
- `docs/plan/PLAN.md` 14 个 M 路线图
- `docs/plan/PROGRESS.md` 状态跟踪
- `docs/kb/REFERENCES.md` 参考资料分层
- `docs/setup.md` 环境与命令

#### 知识库层（单文件多 H2，平面化）
- `docs/kb/knowledge.md` 知识点（papers / libs / concepts / tools 四章）
- `docs/kb/lessons.md` 教训（按时间追加 L1, L2, ...）
- `docs/kb/decisions.md` ADR（本文件）

#### AI 协作层
- `CLAUDE.md` 项目级 AI 常驻记忆
- `.claude/commands/` 5 个 slash 命令（plan / work / review / archive / preflight）

#### Memory 层（CodeFlicker 长期记忆）
- 同步 `update_memory`，关键字 `inferlite`，跨会话可 `search_memory` 检索

### 后果

**优势**：
- 任务卡可独立 review/version/link
- 教训可被多个未来任务卡引用
- 跨会话知识不丢失（Memory 兜底）
- AI 协作纪律有强制 anchor

**成本**：
- 任务卡 ✅ 时要执行额外的 archive 步骤
- 但通过 `/archive` 命令一键完成，边际成本可接受

**替代方案否决**：
- A. 仅文件不入 Memory：跨会话依赖人工 grep
- B. 仅 Memory：人无法 grep / 分享 / 版本化
- C. 不沉淀：每个 M 从零规划，无法复用经验

### 参考
- github/spec-kit
- Anthropic Claude Code Best Practices
- Addy Osmani《My LLM Coding Workflow Going Into 2026》
- docs/kb/REFERENCES.md §AI 协作方法论

---

## ADR-002: 知识库与代码同仓（R1 重构，2026-06-07）

**状态**：Accepted (2026-06-07)

### 背景
初版（ADR-001）把知识库放在工作区根 `~/learning/docs/projects/inferlite/`，与代码仓库 `~/learning/inferlite/` 分离，理由是"工作区级知识库不应该绑死项目仓库"。

实际使用后发现问题：
1. **文件过度零碎**：14 个文件分散在 6 个子目录，单文件 40-80 行
2. **跨仓同步成本**：代码改动与知识库改动不在同一 commit，PR review 看不到知识库变化
3. **AI 读上下文成本高**：要读知识库需多次 `read_file`
4. **搜索体验差**：跨仓 grep 不便

### 决策
**R1 重构**：
- 知识库全部迁回 `inferlite/docs/` 内
- `knowledge / lessons / decisions` 合并为单文件多 H2
- `mainline 草稿区`废除：M 完成时直接读所有任务卡聚合写 summary，不需要中间草稿
- 9 个 slash 命令合并为 5 个

文件数：35 → 12（减 65%）

### 后果
- 改动与代码同 commit，PR review 完整
- AI 一次 `read_file knowledge.md` 拉完所有原子卡
- 全文搜索 cmd+F 即可
- 任务卡仍保留"一卡一文件"（PR 粒度独立）

### 风险与缓解
- 单文件膨胀：估算 100 张知识卡 ≈ 5000 行，仍可控；若超 8000 行再考虑分卷
- 合并冲突：单人项目可忽略；未来多人时按章节拆分即可

### 参考
- ADR-001（被本 ADR 修订知识库位置部分）

---

## 维护规则
- **新增 ADR**：在文件末尾 `## ADR-NNN: <title>`，编号递增（顺序，不复用废弃号）
- **修订 ADR**：原 ADR 加 `**Status**: Superseded by ADR-NNN`，不删原文
- **格式固定**：状态 / 背景 / 决策 / 后果 + "参考"

---

## ADR-003: 分组目录 + 文档可视化（MkDocs Material）

**状态**：Accepted (2026-06-07)

### 背景
ADR-002 R1 重构后 docs/ 平铺 8 个文件，看不出"哪是规划 / 哪是执行 / 哪是知识"。
且纯目录树看不出文件之间的引用关系，需要可视化入口。

### 决策
1. **按"何时读"分 3 组 + 1 独立**：
   - `plan/` 规划层（PLAN/PROGRESS/M\<n\>）
   - `tasks/` 执行层（任务卡）
   - `kb/` 知识层（knowledge/lessons/decisions/REFERENCES）
   - `setup.md` 顶层独立，新人入口
   - 数字前缀天然按阅读顺序排序

2. **`docs/README.md` 作为总入口**：含 mermaid 文档地图，点击节点跳转，GitHub 原生渲染

3. **MkDocs Material 文档站**：
   - `make docs-serve` 本地 http://localhost:8000
   - 左侧 sidebar 按 1/2/3 分组 / 顶部全文搜索 / 暗色模式 / mermaid 自动渲染
   - GitHub Actions `.github/workflows/docs.yml` 自动 deploy 到 gh-pages
   - 仓库 Settings → Pages 启用 gh-pages branch

### 后果
- **文件数不变**（13），但视觉上 3 个文件夹一目了然
- 总入口 README.md + 文档站，新人 30s 看懂结构
- 跨文档关系图谱化，不再"脑补"
- 代价：14 个 dev 依赖（mkdocs + material + mermaid2 + pymdown-extensions），但只在 `--group dev`，运行时无负担

### 风险
- mkdocs-material 8.0+ 后将闭源（见 build warning），但当前 9.x 仍可长期用
- mermaid 复杂图在 GitHub 上可能渲染慢；保持 ≤20 节点

### 参考
- ADR-002（被本 ADR 延伸：在平面化基础上加分组 + 可视化）
