# inferlite 项目常驻记忆（Claude Code / CodeFlicker / 任何 AI 协作者读这里）
#
# 这是项目级 spec，跟 `~/learning/AGENTS.md`（工作区级）配合使用。
# 工作区级管"文件放哪"，项目级管"代码写什么/不写什么"。

## 项目定位
- inferlite = 从零手撕的 LLM 推理引擎学习项目
- 主要参考: nano-vllm（千行体量目标）、rasbt LLMs-from-scratch（教学）、transformers（数值对齐）
- 详见 `docs/1-plan/PLAN.md`（14 个里程碑）、`docs/1-plan/M1.md`（当前 M1 作战地图）

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

## 知识库（双轨制 · R1 平面化 + 分组可视化版）
- **文件**（全在 `inferlite/docs/` 内，按 1/2/3/4 分组）：
  - `docs/README.md` 文档地图（mermaid 跳转图，总入口）
  - `docs/1-plan/{PLAN,PROGRESS,M<n>}.md` 规划层
  - `docs/2-tasks/M<n>-T*.md` 任务卡（一卡一文件）
  - `docs/3-kb/knowledge.md` 知识点（H2: Papers/Libraries/Concepts/Tools）
  - `docs/3-kb/lessons.md` 教训（L1, L2, ...）
  - `docs/3-kb/decisions.md` ADR
  - `docs/3-kb/REFERENCES.md` 参考资料
  - `docs/4-setup.md` 环境 + 仓库结构
- **Memory**：CodeFlicker repos dimension，关键字 `inferlite`
- **可视化**：`make docs-serve` 起本地 MkDocs Material（http://localhost:8000），或 GitHub Pages（push main 自动 deploy）
- 新会话进入项目时先 `search_memory("inferlite")`，再读 `docs/README.md` + `docs/1-plan/PROGRESS.md` + 当前 `M<n>.md`
- 详见 `docs/3-kb/decisions.md` ADR-001 / ADR-002 / ADR-003

## Slash 命令（5 个）
- `/plan <scope>` — 规划（M / T / 调整），含前置调研，自动补 knowledge.md
- `/work <task-id>` — 开任务卡，含前置 knowledge gap 检查，输出作战简报
- `/review <task-id>` — review 已完成的任务卡
- `/archive task <id>` / `/archive milestone M<n>` — 归档（含 lessons + knowledge + summary）
- `/preflight` — 环境健康检查

## 测试纪律
- 每个手写模块必须有 L0 单测 vs `transformers.models.qwen3.modeling_qwen3.*` allclose
- 容差: fp32 1e-5 / fp16/bf16 5e-3
- 12 cases 全绿才能进下一张任务卡
- 详见 `docs/1-plan/M1.md` §7 测试金字塔

## 数值对齐 ground truth
- transformers==5.10.2（已 lock 在 pyproject.toml）
- 所有 ground truth: `from transformers.models.qwen3.modeling_qwen3 import Qwen3*`
- 任何"我觉得应该这样"的写法都先 diff vs transformers

## 网络环境（国内）
- HuggingFace 不可达 → 默认走 ModelScope
- `make preflight` 已配置好
- 详见 `docs/4-setup.md` §4.2 + `docs/3-kb/lessons.md` L2

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
