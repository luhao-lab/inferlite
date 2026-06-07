# inferlite 项目常驻记忆（Claude Code / CodeFlicker / 任何 AI 协作者读这里）
#
# 这是项目级 spec，跟 `~/learning/AGENTS.md`（工作区级）配合使用。
# 工作区级管"文件放哪"，项目级管"代码写什么/不写什么"。

## 项目定位
- inferlite = 从零手撕的 LLM 推理引擎学习项目
- 主要参考: nano-vllm（千行体量目标）、rasbt LLMs-from-scratch（教学）、transformers（数值对齐）
- 详见 `docs/PLAN.md`（14 个里程碑）、`docs/M1.md`（当前 M1 作战地图）

## 角色分工（重要）
- **作者本人**手写所有 `inferlite/**/*.py` 业务代码
- **AI 助手**仅做：
  - Plan / 任务卡设计 / Review
  - 文档撰写（docs/ 下所有 .md）
  - 测试代码（tests/，因为是验证而非学习目标）
  - 脚本（scripts/，工程辅助）
  - CI / 配置（.github/, pyproject.toml, .pre-commit-config.yaml）
- **绝对不要**：AI 直接生成 `inferlite/model/`、`inferlite/engine/` 等核心实现

## 任务推进协议
1. 用户说"开 TX 任务卡"或 `/next-task` → 展开 `docs/tasks/M1-TX-*.md`
2. 用户写代码 + 跑测试 → 贴结果
3. AI review（`/review-card TX`）→ 提改进 → commit
4. 更新 `docs/PROGRESS.md` + `docs/tasks/README.md` 状态列
5. **`/archive-task TX` 双轨沉淀教训**（文件 + Memory）

## 知识库（双轨）
- 文件: `~/learning/docs/projects/inferlite/{lessons,decisions,milestones,benchmarks}/`
- Memory: CodeFlicker repos dimension，关键字 `inferlite`
- 新会话进入项目时先 `search_memory("inferlite")`，再读 `docs/projects/inferlite/README.md` 索引
- 详见 ADR-001（`~/learning/docs/projects/inferlite/decisions/001-spec-driven-workflow.md`）

## Slash 命令
- `/next-task` — 开下一张任务卡
- `/review-card TX` — review 已完成的 TX
- `/preflight-check` — 开工前环境健康检查
- `/archive-task TX` — 任务卡完成后双轨沉淀
- `/archive-milestone Mn` — 里程碑完成归档
- `/plan-next-milestone Mn+1` — 基于知识库规划下一里程碑

## 测试纪律
- 每个手写模块必须有 L0 单测 vs `transformers.models.qwen3.modeling_qwen3.*` allclose
- 容差: fp32 1e-5 / fp16/bf16 5e-3
- 12 cases 全绿才能进下一张任务卡
- 详见 `docs/M1.md` §7 测试金字塔

## 数值对齐 ground truth
- transformers==5.10.2（已 lock 在 pyproject.toml）
- 所有 ground truth: `from transformers.models.qwen3.modeling_qwen3 import Qwen3*`
- 任何"我觉得应该这样"的写法都先 diff vs transformers

## 网络环境（国内）
- HuggingFace 不可达 → 默认走 ModelScope
- `make preflight` 已配置好（commit 5b7fc5e 起）
- 详见 `docs/SETUP.md` §5.4 + `docs/M1.md` §8 坑 #9

## 反模式（NEVER）
- ❌ AI 直接写 `inferlite/model/*.py` 业务代码（侵犯学习目标）
- ❌ 跳过 L0 测试就 commit（违反"每模块对齐"纪律）
- ❌ 一次塞超过 1 张任务卡的代码（违反小步 commit）
- ❌ 在任务卡执行中插入环境调试（违反"地基/算法两频道"）
- ❌ 用 conda base 的 python/pytest（用 `uv run`）

## commit message 规范
- 业务: `feat(model): RMSNorm aligned with Qwen3RMSNorm (T1 done)`
- 测试: `test(rmsnorm): add 12 cases vs reference`
- 文档: `docs(M1): restructure with arch diagram`
- 工程: `chore(ci): add macos matrix`
- 修复: `fix(loader): handle tie_word_embeddings missing key`
