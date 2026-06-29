# inferlite 文档目录

> 从零手写 LLM 推理引擎 · 代码全手敲 · AI 辅助规划/复盘/文档
> GitHub: [luhao-lab/inferlite](https://github.com/luhao-lab/inferlite)

---

## 快速上手

```bash
git clone git@github.com:luhao-lab/inferlite.git && cd inferlite
make setup        # 安装 uv + 同步依赖 + 注册 pre-commit hook
make preflight    # 下载 Qwen3-0.6B（国内走 ModelScope，~5-15 min）
make test         # 跑全部单测，应全绿
```

完成后用 `uv run ...` 而非裸 `python`（自动注入 venv，无需 activate）。

```bash
uv run python -m inferlite.cli "你好"      # 跑推理
uv run pytest tests/unit/test_rmsnorm.py -v
make lint && make fmt && make typecheck
```

> **大坑**：国内 huggingface.co 不可达 → 用 `make preflight`（已配 ModelScope）。
> Makefile 执行行必须 **Tab** 缩进，不能空格。不要用 conda base 的 pytest，用 `uv run pytest`。

---

## 文档地图

### plan/ — 规划层（想了解项目方向/进度时读这里）

| 文件 | 内容 | 什么时候读 |
|------|------|-----------|
| [plan/PLAN.md](./plan/PLAN.md) | 14 个里程碑路线图，每个 M 的目标范围、不做什么、参照项目 | 想了解整个项目要做什么、为什么这么规划 |
| [plan/PROGRESS.md](./plan/PROGRESS.md) | 每个 M 的状态（⬜/🟡/✅）+ 每次任务完成的变更日志 | 每次开工先看：当前做到哪了，下一步是什么 |
| [plan/M1.md](./plan/M1.md) | M1 作战地图：架构图、任务总表（T0~T12）、测试金字塔、完成定义 | M1 推进期开工前读；M1 完成后作为历史参考 |
| [plan/m2-kv-cache-design.md](./plan/m2-kv-cache-design.md) | M2 技术设计：KV Cache 方案调研、ADR 决策、数据流、代码骨架 | M2 推进期开工前读；想理解 KV Cache 设计思路时 |

### tasks/ — 执行层（想看具体怎么做/做到哪了时读这里）

| 文件 | 内容 | 什么时候读 |
|------|------|-----------|
| [tasks/M2-T1~T5](./tasks/) | 当前 M2 的 5 张活跃任务卡，每卡含：算法核心、L0 测试清单、DoD、易踩坑 | 开始写某个模块前读对应任务卡 |
| [tasks/_TEMPLATE.md](./tasks/_TEMPLATE.md) | 任务卡 7 字段模板（`/work` 命令自动填充） | 新建任务卡时参考格式 |
| [tasks/M1-archive/](./tasks/M1-archive/) | M1 全部 12 张已完成任务卡（T0~T12） | 想回顾 M1 某个模块的实现思路、踩坑记录时 |

### kb/ — 知识层（想查知识/防坑/理解模块时读这里）

| 文件 | 内容 | 什么时候读 |
|------|------|-----------|
| [kb/knowledge.md](./kb/knowledge.md) | 知识卡片库：Papers（论文）/ Libraries（框架 API）/ Concepts（核心概念）/ Tools（工程工具）/ ADR（架构决策）/ 参考资料 | 开始任务卡前查前置知识；调研新知识后追加 |
| [kb/lessons.md](./kb/lessons.md) | 踩坑教训 L1~L4，叙事性，有现场感：现象 → 根因 → 解法 → 适用范围 | 遇到奇怪 bug 先来这里查；完成任务卡后追加新教训 |
| [kb/blueprints.md](./kb/blueprints.md) | 模块契约卡片：每个模块的接口签名、设计意图、踩坑、跨 M 依赖关系 | 改某个模块前先看它的 blueprint；M 归档时更新 |

---

## 工作流程（文档协作关系）

这个项目用 **spec-driven** 工作流：先写文档，再写代码，代码和文档同 commit。

### 里程碑生命周期

```
1. /plan M<n>          AI 调研 + 产出设计文档 plan/m<n>-design.md + 任务卡骨架
        ↓
2. /work M<n>-T<x>     AI 输出作战简报（前置知识检查 + 接口签名 + 测试清单）
        ↓
3. 作者写代码           inferlite/**/*.py（AI 不写业务代码）
        ↓
4. 作者跑测试           uv run pytest → 贴结果
        ↓
5. /review M<n>-T<x>   AI review 代码质量 → 建议修改
        ↓
6. /archive task T<x>  AI 沉淀：更新 kb/lessons.md + kb/knowledge.md + kb/blueprints.md
        ↓               更新 plan/PROGRESS.md 勾选任务 → commit
7. （重复 2-6 直到本 M 所有任务卡完成）
        ↓
8. /archive milestone M<n>   AI 写 Summary + 更新 PROGRESS.md → git tag
```

### 各文档的更新时机

| 文档 | 谁更新 | 什么时候更新 |
|------|--------|-------------|
| `plan/PLAN.md` | AI | 新开里程碑、调整路线时（/plan 命令） |
| `plan/PROGRESS.md` | AI | 每张任务卡 ✅ 时（/archive 命令自动更新） |
| `plan/M<n>.md` | AI | 新开里程碑时新建；里程碑归档时追加 Summary |
| `plan/m<n>-design.md` | AI | 里程碑启动时创建，推进期持续完善 |
| `tasks/M<n>-T<x>.md` | AI | /work 开卡时创建；/archive 时末尾追加完成总结 |
| `kb/knowledge.md` | AI | 每次 /archive 后追加新知识卡片 |
| `kb/lessons.md` | AI | 每次 /archive 后追加新教训 |
| `kb/blueprints.md` | AI | 每次 /archive 后更新相关模块的契约 |

### PROGRESS.md vs M\<n\>.md 的区别

- **PROGRESS.md**：横向视图，跨所有里程碑，一行 = 一个 M 的状态。看"现在做到哪了"。
- **M\<n\>.md**：纵向视图，单个里程碑的完整作战细节（架构图 / 任务表 / 测试金字塔）。M 结束后冻结，变成历史档案。

---

## 仓库结构

```
inferlite/
├── CLAUDE.md              # AI 协作约定（项目级常驻记忆，新会话必读）
├── Makefile               # 任务运行器（make help 列出全部目标）
├── pyproject.toml         # Python 项目 + 依赖声明
├── uv.lock                # 依赖锁定（commit 进 git）
├── .claude/commands/      # 5 个 slash 命令（plan/work/review/archive/preflight）
│
├── docs/                  # 本目录（spec + 知识库）
│
├── inferlite/             # 主 Python 包（作者手写，AI 不写这里）
│   ├── model/             # RMSNorm / Attention / RoPE / DecoderLayer / Qwen3
│   ├── engine/            # EngineCore / generate loop
│   ├── sampler/           # GreedySampler
│   └── cli.py
│
├── tests/
│   ├── unit/              # L0 单测（vs transformers ground truth，allclose）
│   ├── integration/       # L1-L2 集成测试
│   └── e2e/               # L3 端到端
│
└── scripts/
    ├── setup.sh           # 一键安装（make setup 调用）
    └── preflight.py       # 开工前体检（make preflight 调用）
```

---

## 当前进度

M1 Qwen3 单序列推理 🟡 → **M2 KV Cache ✅** → M3 Continuous Batching ⬜

详见 [plan/PROGRESS.md](./plan/PROGRESS.md)

---

## 文档站

```bash
make docs-serve    # 本地 http://localhost:8000（侧栏导航 + 全文搜索 + 暗色模式）
make docs-deploy   # 部署到 GitHub Pages
```
