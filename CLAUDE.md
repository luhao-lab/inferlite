# inferlite 项目常驻记忆（Claude Code / CodeFlicker / 任何 AI 协作者读这里）
#
# 这是项目级 spec，跟 `~/learning/AGENTS.md`（工作区级）配合使用。
# 工作区级管"文件放哪"，项目级管"代码写什么/不写什么"。

## 项目定位
- inferlite = 从零手撕的 LLM 推理引擎学习项目
- 主要参考: nano-vllm（千行体量目标）、rasbt LLMs-from-scratch（教学）、transformers（数值对齐）
- 详见 `docs/plan/PLAN.md`（14 个里程碑）、`docs/plan/M1.md`（当前 M1 作战地图）

## 角色分工（重要）
- **作者本人**手写所有 `inferlite/**/*.py` 业务代码
- **AI 助手**仅做：
  - Plan / 任务卡设计 / Review
  - 文档撰写（docs/ 下所有 .md）
  - 测试代码（tests/，因为是验证而非学习目标）
  - 脚本（scripts/，工程辅助）
  - CI / 配置（.github/, pyproject.toml, .pre-commit-config.yaml）
- **绝对不要**：AI 直接生成 `inferlite/model/`、`inferlite/engine/` 等核心实现

## 任务推进协议（R1 简化版 · 5 命令）
1. **新 M 起点**：`/plan M<n>` → 自动调研 + 产出 `docs/M<n>.md` + 任务卡骨架
2. **开任务卡**：`/work T<x>` → 自动检前置 knowledge + 输出作战简报
3. **用户写代码** + 跑测试 → 贴结果
4. **AI review**：`/review T<x>` → 提改进 → commit
5. **任务卡归档**：`/archive task T<x>` → 沉淀 lessons + knowledge + 更新状态
6. **里程碑归档**：`/archive milestone M<n>` → 写 Summary + tag
7. **环境体检**：随时 `/preflight`

## 新会话启动顺序（必须执行）
1. `search_memory("inferlite")` — 拉取跨会话长期记忆
2. 读 `CLAUDE.md`（本文件）— 角色分工 + 文件清单 + 反模式
3. 读 `docs/plan/PROGRESS.md` — 当前进度，确认做到哪了
4. 读当前 M 的设计文档（如 `docs/plan/m2-kv-cache-design.md`）— 理解方案

## 文件清单与更新触发器

> AI 每次 `/archive` 时必须逐项对照此表，确保没有文件被遗漏更新。

| 文件 | 功能（一句话） | 触发更新的事件 | 更新方式 |
|------|--------------|--------------|--------|
| `CLAUDE.md` | AI 常驻规范，新会话必读 | 工作流变化、文件结构变化 | 手动 |
| `docs/README.md` | 文档地图 + 快速上手 + 工作流说明 | 新增/删除文件、工作流变化 | 手动 |
| `docs/plan/PLAN.md` | 14 个里程碑路线图 | 调整路线、新开 M | `/plan` 命令 |
| `docs/plan/PROGRESS.md` | 跨 M 进度状态 + 变更日志 | 每张任务卡 ✅ | `/archive task` |
| `docs/plan/M<n>.md` | 单 M 作战地图（架构/任务/测试） | 新开 M 时新建；M 归档时追加 Summary | `/plan` + `/archive milestone` |
| `docs/plan/m<n>-design.md` | 单 M 技术设计文档 | 里程碑启动时创建 | `/plan` 命令 |
| `docs/tasks/M<n>-T<x>.md` | 任务卡（算法/测试/DoD/坑） | `/work` 开卡；`/archive` 时追加完成总结 | `/work` + `/archive task` |
| `docs/kb/knowledge.md` | 知识卡片（Papers/Libs/Concepts/Tools/ADR/参考资料） | 每次任务归档后追加新卡 | `/archive task` |
| `docs/kb/lessons.md` | 踩坑教训（叙事性，L1~Ln） | 每次任务归档后追加新坑 | `/archive task` |
| `docs/kb/blueprints.md` | 模块契约（接口/踩坑/跨M依赖） | 每次任务归档后更新相关模块 | `/archive task` |
| `mkdocs.yml` | 文档站导航 nav | 新增/删除/移动文档文件 | 手动 |
| 根 `README.md` | GitHub 首页（项目介绍 + 进度） | M 归档、进度大变 | `/archive milestone` |

**Memory**：CodeFlicker repos dimension，关键字 `inferlite`，`update_memory` 在 `/archive` 时同步。

## Slash 命令（5 个）
- `/plan <scope>` — 规划（M / T / 调整），含前置调研，自动补 knowledge.md
- `/work <task-id>` — 开任务卡，含前置 knowledge gap 检查，输出作战简报
- `/review <task-id>` — review 已完成的任务卡
- `/archive task <id>` / `/archive milestone M<n>` — 归档（含 lessons + knowledge + blueprints + summary）
- `/preflight` — 环境健康检查

## 测试纪律
- 每个手写模块必须有 L0 单测 vs `transformers.models.qwen3.modeling_qwen3.*` allclose
- 容差: fp32 1e-5 / fp16/bf16 5e-3
- 12 cases 全绿才能进下一张任务卡
- 详见 `docs/plan/M1.md` §7 测试金字塔

## 数值对齐 ground truth
- transformers==5.10.2（已 lock 在 pyproject.toml）
- 所有 ground truth: `from transformers.models.qwen3.modeling_qwen3 import Qwen3*`
- 任何"我觉得应该这样"的写法都先 diff vs transformers

## 网络环境（国内）
- HuggingFace 不可达 → 默认走 ModelScope
- `make preflight` 已配置好
- 详见 `docs/README.md` §大坑 + `docs/kb/lessons.md` L2

## 反模式（NEVER）
- ❌ AI 直接写 `inferlite/model/*.py` 业务代码（侵犯学习目标）
- ❌ 跳过 L0 测试就 commit（违反"每模块对齐"纪律）
- ❌ 一次塞超过 1 张任务卡的代码（违反小步 commit）
- ❌ 在任务卡执行中插入环境调试（违反"地基/算法两频道"，见 lessons.md L3）
- ❌ 用 conda base 的 python/pytest（用 `uv run`）
- ❌ 把 knowledge 卡拆成新文件（违反 ADR-002 平面化）

## commit message 规范
- 业务: `feat(model): RMSNorm aligned with Qwen3RMSNorm (T1 done)`
- 测试: `test(rmsnorm): add 12 cases vs reference`
- 文档: `docs(M1): restructure with arch diagram`
- 工程: `chore(ci): add macos matrix`
- 修复: `fix(loader): handle tie_word_embeddings missing key`
