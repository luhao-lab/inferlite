# Setup — 环境、命令、仓库结构

> 一站式：`make setup` 一键就绪；`uv` 管依赖；`make` 管任务。

---

## 1. 一键安装

```bash
git clone git@github.com:luhao-lab/inferlite.git
cd inferlite
make setup        # 装 uv + sync 依赖 + 健康检查
make preflight    # 下载 Qwen3-0.6B（CN: ~5-15 min）+ 跑通推理
make test         # 跑测试，应全绿
```

完成后无需 `source .venv/bin/activate`；所有命令用 `uv run ...`（自动注入 venv）。

---

## 2. 常用命令

```bash
# 一次性
make setup
make preflight                        # 下载模型 + 跑通推理（开 M1 前必跑）

# 日常
make sync                             # 重新同步依赖
make test                             # 跑全部测试
make test-fast                        # 跳过 @pytest.mark.slow
make lint                             # ruff 检查
make fmt                              # ruff 格式化
make typecheck                        # mypy
make clean                            # 清 .venv 和缓存
make help                             # 列出所有目标

# 加依赖
uv add "torch>=2.5"                   # 主依赖
uv add --dev pytest-xdist             # dev 依赖
uv lock --upgrade transformers        # 升级单包

# 跑 CLI / 单文件
uv run python -m inferlite.cli "你好"
uv run pytest tests/unit/test_rmsnorm.py -v
```

工具链详细介绍（uv / make / ruff / pre-commit 是什么）见 `docs/kb/knowledge.md` → Tools。

---

## 3. 仓库结构

```
inferlite/
├── CLAUDE.md              # AI 协作约定（项目级常驻记忆）
├── README.md              # 项目首页
├── Makefile               # 任务运行器
├── pyproject.toml         # Python 项目 + 依赖声明
├── uv.lock                # 依赖锁定（commit 进 git）
├── .pre-commit-config.yaml
├── .github/workflows/tests.yml      # CI: ubuntu+macos × py3.12
├── .claude/commands/      # 5 个 slash 命令
│
├── docs/                  # 全部文档（spec + 知识库）
│   ├── PLAN.md            # 14 个 M 路线图
│   ├── PROGRESS.md        # 状态跟踪
│   ├── REFERENCES.md      # 参考资料分层
│   ├── M1.md ...          # 单 M 作战地图（M 完成后追加 Summary 段）
│   ├── tasks/
│   │   ├── README.md
│   │   ├── _TEMPLATE.md
│   │   └── M<N>-T<X>-*.md # 任务卡（一卡一文件）
│   ├── knowledge.md       # 知识点（papers / libs / concepts / tools 四章）
│   ├── lessons.md         # 教训（L1, L2, ...）
│   ├── decisions.md       # ADR
│   └── setup.md           # 本文件
│
├── inferlite/             # 主 Python 包
│   ├── __init__.py
│   ├── model/             # T1-T8 算法实现
│   ├── engine/            # T9-T10 引擎
│   ├── sampler/           # T9 采样
│   ├── server/            # M11 服务化
│   └── utils/
│
├── tests/
│   ├── unit/              # L0 单测（vs transformers ground truth）
│   ├── integration/       # L1-L2 集成
│   └── e2e/               # L3 端到端
│
└── scripts/
    ├── setup.sh           # 一键安装（被 make setup 调用）
    └── preflight.py       # 跑前体检（make preflight）
```

### 文件作用速查

| 路径 | 作用 | 何时会变 |
| --- | --- | --- |
| `pyproject.toml` | 项目元数据 + 依赖声明 + ruff/pytest 配置 | 加依赖 / 改配置 |
| `uv.lock` | 依赖版本锁定 | `uv add/sync/lock` 自动 |
| `Makefile` | 任务别名 | 加新任务 |
| `CLAUDE.md` | AI 协作约定 | 工作流变化 |
| `docs/plan/PLAN.md` | 14 个 M 总览，**单一来源** | 路线图调整 |
| `docs/plan/PROGRESS.md` | 实时状态 | 每张卡 ✅ 时 |
| `docs/M<N>.md` | 单 M 作战地图 | 开新 M 时新建 |
| `docs/tasks/M<N>-T<X>.md` | 任务卡 | 开新卡 / 卡完成 |
| `docs/kb/knowledge.md` | 知识点（单文件多 H2） | 调研时追加章节 |
| `docs/kb/lessons.md` | 教训 | 任务卡 ✅ 时追加 |
| `docs/kb/decisions.md` | ADR | 重大决策 |

---

## 4. 大坑预警

### 4.1 Mac MPS + PyTorch 版本
- M1-M5 全程 `torch.float32` + MPS；不要碰 fp16/bf16（MPS 后端某些算子有坑）
- transformers 必须 ≥ 5.10

### 4.2 国内网络下载 Qwen3-0.6B

`huggingface.co` 在国内不可达。详见 `docs/kb/lessons.md` L2。

**结论**：默认走 ModelScope（`make preflight` 已配好）。

### 4.3 Makefile 必须 Tab 缩进

`Makefile` 执行命令行**必须 Tab**，不能空格。否则 `*** missing separator. Stop.`

### 4.4 不要用 conda base 的 python/pytest
- 用 `uv run pytest` 而非裸 `pytest`
- 避免与 conda 环境冲突

更多坑见 `docs/kb/lessons.md`。

---

## 5. 维护约定

往仓库根加新顶层文件/目录时：
1. 更新本文件 §3 "仓库结构" + "文件作用速查"
2. 如涉及新工具/库，在 `docs/kb/knowledge.md` → Tools 加章节

---

## 6. 文档站（MkDocs Material）

本地浏览 / 全文搜索 / mermaid 渲染：

```bash
make docs-serve        # http://localhost:8000
make docs-build        # 输出到 site/（已 .gitignore）
make docs-deploy       # gh-pages（需先开仓库 Settings → Pages → gh-pages branch）
```

技术栈：
- `mkdocs-material` 主题（侧栏 nav / 顶部搜索 / 暗色模式）
- `mkdocs-mermaid2-plugin` 自动渲染 mermaid 流程图
- 配置：`mkdocs.yml`

新增文档时无需改 `mkdocs.yml`（已配置自动收录），除非要调整 nav 顺序或分组。
