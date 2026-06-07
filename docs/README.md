---
hide:
  - navigation
  - toc
---

<div class="geek-hero" markdown>

# inferlite

<p class="tagline">从零手撕的 LLM 推理引擎学习项目 · L0-aligned with transformers</p>

<div class="badges" markdown>

![CI](https://img.shields.io/github/actions/workflow/status/luhao2013/inferlite/tests.yml?branch=main&label=CI&style=flat-square&color=26c6da)
![Tests](https://img.shields.io/badge/tests-12%2F12-5dff9b?style=flat-square)
![Python](https://img.shields.io/badge/python-3.12-26c6da?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-94a3b8?style=flat-square)

</div>

<div class="actions" markdown>

[:octicons-rocket-16: &nbsp;立即开始](./4-setup.md){ .primary }
[:octicons-graph-16: &nbsp;查看进度](./1-plan/PROGRESS.md){ .secondary }
[:fontawesome-brands-github: &nbsp;GitHub](https://github.com/luhao2013/inferlite){ .secondary }

</div>

</div>

## 文档导航

<div class="card-grid" markdown>

<a class="card" href="./1-plan/PLAN.md"><span class="icon">:octicons-milestone-16:</span><span class="title">1-plan/</span><span class="desc">14 个里程碑路线图、当前作战地图 M1、整体进度跟踪。</span><span class="meta">PLAN · PROGRESS · M&lt;n&gt;</span></a>

<a class="card" href="./2-tasks/README.md"><span class="icon">:octicons-checklist-16:</span><span class="title">2-tasks/</span><span class="desc">任务卡七字段（前置 / 边界 / 验收 / 风险 / 完成总结），一卡一文件。</span><span class="meta">M1-T0 · M1-T1 ✓ · M1-T2</span></a>

<a class="card" href="./3-kb/knowledge.md"><span class="icon">:octicons-book-16:</span><span class="title">3-kb/</span><span class="desc">知识库：论文 / 库 / 概念 / 工具 卡片化总结，踩坑教训 + ADR 决策记录。</span><span class="meta">knowledge · lessons · decisions</span></a>

<a class="card" href="./4-setup.md"><span class="icon">:octicons-tools-16:</span><span class="title">4-setup.md</span><span class="desc">一键 uv install、常用 make 命令、仓库结构速查、大坑预警。</span><span class="meta">uv · make · ruff · pytest</span></a>

</div>

## 文档关系图

```mermaid
flowchart TD
    USER([新会话进入]):::user --> README[docs/README.md<br/>本页 · 总入口]:::entry
    README -->|新人| SETUP[4-setup.md<br/>环境 + 仓库结构]:::setup
    README -->|看全貌| PLAN[1-plan/PLAN.md<br/>14 个 M 路线图]:::plan
    README -->|开工| PROGRESS[1-plan/PROGRESS.md<br/>状态跟踪]:::plan
    README -->|查知识| KB[3-kb/knowledge.md<br/>Papers/Libs/Concepts/Tools]:::kb

    PROGRESS --> M1[1-plan/M1.md<br/>当前作战地图]:::plan
    M1 --> T0P[2-tasks/M1-T0<br/>ModelConfig]:::tasks
    M1 --> T1[2-tasks/M1-T1<br/>RMSNorm ✓]:::tasks
    M1 --> T2[2-tasks/M1-T2<br/>SwiGLU]:::tasks

    T0P -.前置阅读.-> KB
    T1 -.沉淀.-> LESSONS[3-kb/lessons.md]:::kb
    T1 -.沉淀.-> KB
    PLAN -.方法论.-> DEC[3-kb/decisions.md<br/>ADR-001/002/003]:::kb
    KB -.原始链接.-> REF[3-kb/REFERENCES.md]:::kb

    classDef user fill:#1a1f2e,stroke:#7fdbca,stroke-width:2px,color:#fff3b0
    classDef entry fill:#0a2540,stroke:#26c6da,stroke-width:2px,color:#26c6da
    classDef setup fill:#0d2818,stroke:#5dff9b,color:#5dff9b
    classDef plan fill:#291638,stroke:#b388ff,color:#b388ff
    classDef tasks fill:#1a1538,stroke:#7c4dff,color:#bb9bff
    classDef kb fill:#3a1a0d,stroke:#ffb74d,color:#ffb74d

    click README "./"
    click SETUP "./4-setup.md"
    click PLAN "./1-plan/PLAN.md"
    click PROGRESS "./1-plan/PROGRESS.md"
    click M1 "./1-plan/M1.md"
    click T0P "./2-tasks/M1-T0-ModelConfig.md"
    click T1 "./2-tasks/M1-T1-RMSNorm.md"
    click T2 "./2-tasks/M1-T2-SwiGLU.md"
    click KB "./3-kb/knowledge.md"
    click LESSONS "./3-kb/lessons.md"
    click DEC "./3-kb/decisions.md"
    click REF "./3-kb/REFERENCES.md"
```

## 三类文档

| 分类           | 时间维度       | 写作时机                | 文件                                                                |
| -------------- | -------------- | ----------------------- | ------------------------------------------------------------------- |
| **1-plan/**    | 未来 / 当前    | 规划 + 持续更新         | `PLAN.md` `PROGRESS.md` `M<n>.md`                                   |
| **2-tasks/**   | 当前           | 开工时写、完成时收尾    | `M<n>-T<x>-*.md`                                                    |
| **3-kb/**      | 过去（沉淀）   | 任务结束后归档          | `knowledge.md` `lessons.md` `decisions.md` `REFERENCES.md`          |
| **4-setup.md** | 长期常驻       | 项目稳定后不常变        | `4-setup.md`                                                        |

## 三条阅读路径

=== ":octicons-rocket-16: 新人 onboarding"

    1. 你正在看的 `README.md` — 知道有哪些文档
    2. [4-setup.md](./4-setup.md) — 5 分钟跑起项目
    3. [1-plan/PLAN.md](./1-plan/PLAN.md) — 看整体路线
    4. [3-kb/decisions.md](./3-kb/decisions.md) — 理解为什么这么做

=== ":octicons-checklist-16: 接任务"

    1. [1-plan/PROGRESS.md](./1-plan/PROGRESS.md) — 找下一张 :material-checkbox-blank-outline: 任务
    2. [2-tasks/](./2-tasks/README.md) 中读对应任务卡 7 字段
    3. [3-kb/knowledge.md](./3-kb/knowledge.md) — 按"前置"段查相关章节

=== ":octicons-history-16: 复盘"

    1. [3-kb/lessons.md](./3-kb/lessons.md) — 全部踩坑
    2. 当前 `M<n>.md` 末尾 Summary
    3. 任务卡末尾"完成总结"段

## 本地预览

<div class="termy" markdown>

```bash
$ make docs-serve              # http://localhost:8000
$ make docs-build              # → site/
$ make docs-deploy             # gh-pages
```

</div>

---

!!! tip "AI 协作"
    本仓库使用 spec-driven workflow，AI 协作约定见仓库根 `CLAUDE.md`。
    新会话进入前建议先 `search_memory("inferlite")`。
