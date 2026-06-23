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
uv run python -m inferlite.cli "你好"     # 跑推理
uv run pytest tests/unit/test_rmsnorm.py -v
make lint && make fmt && make typecheck
```

> **大坑**：国内 huggingface.co 不可达 → 用 `make preflight`（已配 ModelScope）。
> Makefile 执行行必须 **Tab** 缩进，不能空格。不要用 conda base 的 pytest，用 `uv run pytest`。

---

## 文档目录

```
docs/
├── README.md          ← 本文件：快速上手 + 目录导航
│
├── plan/              ← 「规划层」记录做什么、为什么、进度到哪了
│   ├── PLAN.md            14 个里程碑路线图（项目全貌）
│   ├── PROGRESS.md        每个 M 的进度状态 + 变更日志
│   ├── M1.md              M1 作战地图（架构图、任务总表、测试金字塔）
│   └── m2-kv-cache-design.md   M2 技术设计文档（方案调研、ADR、数据流）
│
├── tasks/             ← 「执行层」每张卡是一次 PR 粒度的作战单元
│   ├── _TEMPLATE.md       7 字段任务卡模板（/work 命令自动填充）
│   ├── M2-T1~T5.md        当前活跃任务卡（M2 KV Cache 阶段）
│   └── M1-archive/        M1 已完成任务卡（历史档案）
│
├── kb/                ← 「知识层」沉淀可复用的知识，防止经验流失
│   ├── knowledge.md       知识卡片：Papers / Libraries / Concepts / Tools / ADR / 参考资料
│   ├── lessons.md         踩坑教训（L1~L4，叙事性，按时间追加）
│   └── blueprints.md      模块契约（每个模块的接口、踩坑、跨 M 依赖）
│
└── _assets/           ← MkDocs 网站 UI 资产（非文档内容）
    ├── javascripts/
    └── stylesheets/
```

---

## 仓库结构

```
inferlite/
├── CLAUDE.md              # AI 协作约定（项目级常驻记忆）
├── Makefile               # 任务运行器（make help 列出全部目标）
├── pyproject.toml         # Python 项目 + 依赖声明
├── uv.lock                # 依赖锁定（commit 进 git）
├── .pre-commit-config.yaml
├── .github/workflows/     # CI: ubuntu+macos × py3.12
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
│   ├── unit/              # L0 单测（vs transformers ground truth）
│   ├── integration/       # L1-L2 集成测试
│   └── e2e/               # L3 端到端
│
└── scripts/
    ├── setup.sh           # 一键安装（make setup 调用）
    └── preflight.py       # 开工前体检（make preflight 调用）
```

---

## 阅读指引

**开始一个里程碑**：`plan/PROGRESS.md` → 当前 M 的设计文档 → `tasks/M*-T*.md` 逐卡推进

**接手一张任务卡**：任务卡（算法核心 + 测试清单 + DoD）→ `kb/knowledge.md`（前置知识）→ `kb/blueprints.md`（模块契约）

**踩坑 / 复盘**：`kb/lessons.md` 查已有的坑；完成后往 `kb/lessons.md` 和 `kb/knowledge.md` 追加一条

---

## 当前进度

M1 Qwen3 单序列推理 ✅ → **M2 KV Cache 进行中** → M3 Continuous Batching ⬜

详见 [plan/PROGRESS.md](./plan/PROGRESS.md)

---

## 文档站

```bash
make docs-serve    # 本地 http://localhost:8000（侧栏导航 + 全文搜索 + 暗色模式）
make docs-deploy   # 部署到 GitHub Pages
```
