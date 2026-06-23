---
description: 任务卡开工前的环境健康检查（包骨架、preflight、依赖、CI 状态）
---

执行以下检查并以 checklist 形式输出：

## 1. 包骨架
```bash
[ -f inferlite/__init__.py ] && uv run python -c "import inferlite; print('inferlite OK')"
```

## 2. 依赖锁定
```bash
uv sync --frozen 2>&1 | tail -3
```

## 3. 现有测试
```bash
uv run pytest tests/ -q --tb=no 2>&1 | tail -5
```

## 4. preflight（可选，慢）
若用户加了 `--full` 参数，跑：
```bash
uv run python scripts/preflight.py
```

## 5. CI 最近一次状态
```bash
gh run list --limit 1 --workflow tests.yml 2>/dev/null || echo "gh CLI 未安装，去 https://github.com/luhao-lab/inferlite/actions 看"
```

## 6. git 状态
```bash
git status --short
git log --oneline -3
```

输出格式（绿 ✅ / 红 ❌ 标注）：
```
[✅] 包骨架: inferlite import OK
[✅] 依赖: 64 packages locked
[✅] 测试: 12 passed
[⏸️] preflight: 未跑（加 --full 触发）
[✅] CI: 最近一次 success on main
[⚠️] git: 有 2 个未提交改动 (M docs/plan/M1.md, ?? new.md)

可以开工: 是 / 否
建议下一步: ...
```
