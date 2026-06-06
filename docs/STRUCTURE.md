# 仓库结构说明

> 列出 `inferlite` 仓库当前所有顶层文件 / 目录及其作用。每次新增文件请在这里加一行解释，避免新人/未来的自己看不懂。

---

## 当前文件树

```
inferlite/
├── .gitignore           # git 忽略规则
├── LICENSE              # MIT 许可证
├── Makefile             # 任务运行器（make setup / test / lint）
├── README.md            # 项目首页
├── pyproject.toml       # Python 项目配置 + 依赖声明
├── uv.lock              # 依赖版本锁定（uv 自动生成）
├── docs/                # 文档目录
│   ├── M1.md            # M1 里程碑 brief（资料/骨架/测试/坑）
│   ├── PLAN.md          # 完整路线图（4 层抽象 / 7 Protocol / 14 个 M）
│   ├── PROGRESS.md      # 实时里程碑进度跟踪
│   ├── SETUP.md         # 环境与日常命令详解（uv/make 是什么）
│   └── STRUCTURE.md     # 本文件
└── scripts/
    └── setup.sh         # 一键安装脚本（被 make setup 调用）
```

---

## 文件逐项解释

### 根目录

| 文件 | 作用 | 何时会变 | 是否 commit |
| --- | --- | --- | --- |
| `.gitignore` | 告诉 git "哪些文件不进版本库"。当前忽略 `__pycache__/`、`.venv/`、`*.safetensors`、bench 输出等 | 加入新工具/产物时 | ✅ |
| `LICENSE` | MIT 协议文本。允许任何人自由使用/修改/商用，只需保留版权声明 | 几乎不变 | ✅ |
| `Makefile` | **任务运行器**。定义 `make setup`、`make test` 这些短命令对应跑什么 shell 命令。详见 [SETUP.md §2.2](SETUP.md) | 加新任务时 | ✅ |
| `README.md` | 项目首页：定位、路线图速览、链接到详细文档 | 阶段性大变更 | ✅ |
| `pyproject.toml` | **Python 项目元数据 + 依赖声明的标准格式**。详见下面 §A | 加/改依赖、配置 ruff/pytest 时 | ✅ |
| `uv.lock` | **依赖版本锁定文件**，记录每个包的精确版本和 hash。详见下面 §B | `uv add/sync/lock` 时自动更新 | ✅ |

### docs/

| 文件 | 作用 | 何时会变 |
| --- | --- | --- |
| `PLAN.md` | 完整路线图。**单一来源**：4 层抽象、7 Protocol、L0–L3 验证、Bench 三件套、M1–M15+ 的所有里程碑设计 | 路线图调整时 |
| `PROGRESS.md` | 实时进度。每个 M 完成时勾选状态、写 tag、贴文章链接 | 每个 M 完成 |
| `M<N>.md` | 单个里程碑的详细 brief（必读资料、文件骨架、Protocol 签名、L0 测试清单、易踩坑） | 开新 M 时新增 |
| `SETUP.md` | 环境与日常命令详解（uv/make 科普 + cheatsheet + 踩坑） | 工具链变化时 |
| `STRUCTURE.md` | 本文件 —— 仓库结构地图 | 加新顶层文件/目录时 |

### scripts/

| 文件 | 作用 |
| --- | --- |
| `setup.sh` | 一键安装脚本，被 `make setup` 调用。检测 uv → 缺则装 → `uv sync` → 健康检查 |

---

## §A. pyproject.toml 详解

**它是什么**：Python 官方推荐的「项目元数据 + 依赖声明 + 工具配置」**单一配置文件**（PEP 518/621），**取代了**老的 `setup.py`/`setup.cfg`/`requirements.txt`/`tox.ini` 等一堆碎文件。

**它由谁读**：

- `uv` / `pip` 读 `[project]` 段 → 知道要装什么依赖
- `hatchling`（build 后端）读 `[build-system]` → 知道如何打 wheel
- `ruff` 读 `[tool.ruff]` → 知道 lint 规则
- `pytest` 读 `[tool.pytest.ini_options]` → 知道测试目录

**结构剖析**（对照仓库当前 `pyproject.toml`）：

```toml
[project]                         # ① 项目身份证（PEP 621 标准段）
name = "inferlite"                #    包名
version = "0.1.0"                 #    版本号
requires-python = ">=3.11"        #    Python 版本约束
dependencies = [                  #    运行时依赖（pip install 时会装）
    "torch>=2.4",
    "transformers>=4.51",
    ...
]

[dependency-groups]               # ② 开发依赖组（PEP 735）
dev = ["pytest", "ruff", "mypy"]  #    `uv sync --group dev` 时才装

[build-system]                    # ③ 打 wheel 用的工具链
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]                       # ④ ruff 配置
line-length = 100

[tool.pytest.ini_options]         # ⑤ pytest 配置
testpaths = ["tests"]
```

**没有它会怎样**：`uv sync` 不知道要装什么，`pip install -e .` 报错说找不到 metadata。

**改它的时机**：
- 加新依赖时 → 用 `uv add foo`，会自动写 `pyproject.toml` + 更新 `uv.lock`（**不要手改这里再忘了 lock**）
- 改 ruff/pytest 配置时 → 直接编辑 `[tool.ruff]` / `[tool.pytest.ini_options]`

---

## §B. uv.lock 详解

**它是什么**：`uv` 自动生成的「依赖版本快照」。`pyproject.toml` 说"我要 `torch>=2.4`"，`uv.lock` 说"实际锁的是 `torch==2.7.0` + 全部传递依赖（numpy==2.0.1, sympy==1.13.3, ...）+ 每个 wheel 的 sha256"。

**为什么要 commit**：
- `pyproject.toml` 只声明范围（`>=2.4`），明天 torch 出 2.8 你一 `uv sync` 就拉到新版，**这周能跑下周不能跑** —— 经典"我本地能跑你那不能跑"
- 有 `uv.lock` 后 `uv sync` 会**严格按 lock 装**，跨机器、跨时间 100% 一致

**类比**：
- `pyproject.toml` ≈ 餐厅菜单（"我要一份米饭，米数量 ≥ 100 粒"）
- `uv.lock` ≈ 上次那顿饭的精确照片（"上回那碗米饭：东北珍珠米 134 粒，水温 92°C"）
- 餐厅说菜单不变就行；想严格复现上次那顿饭，得拿照片

**什么时候更新**：
- `uv add foo` → 加依赖时
- `uv sync` 第一次 → 没有 lock 就生成
- `uv lock --upgrade foo` → 显式升级单个包
- **绝不手改 `uv.lock`**，让 uv 管

**和 .gitignore 的关系**：库（library）通常不 commit lock；**应用（application）必须 commit lock**。inferlite 是应用 → commit。

---

## 维护约定

> 只要往仓库根加新文件 / 新目录，就**同步更新本文件**的"当前文件树"+"逐项解释"两节。

未来 M1 开始后会新增的目录（顶层）：

| 路径 | 作用 |
| --- | --- |
| `inferlite/` | 主 Python 包（M1 起开始有） |
| `tests/` | 测试目录（unit/module/e2e/invariant） |
| `bench/` | benchmark 脚本（M5 起） |
| `examples/` | 用法示例（M5 起，可选） |
