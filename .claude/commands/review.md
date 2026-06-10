---
description: Review 指定任务卡的实现质量。用法：/review-card T2
argument-hint: <task-id>，例如 T2
---

针对用户指定的任务卡 `$ARGUMENTS`（如 T2），执行以下 review：

## 1. 测试覆盖
- 跑 `uv run pytest tests/unit/test_<module>.py -v` 确认全绿
- 检查 L0 测试是否覆盖到 `docs/1-plan/M1.md` §6.X 列出的所有 cases
- 检查 vs transformers ground truth 的 atol/rtol 是否合理

## 2. 实现 review
读 `inferlite/model/*.py` 中对应的新增类，检查：
- 是否遵循 `docs/1-plan/M1.md` §6.X 的算法核心
- shape / dtype 处理是否正确（特别是 fp16/bf16 升 fp32 的边界）
- 命名是否与社区/transformers 一致
- 是否有该任务卡 §6.X "坑" 列表中的反模式

## 3. 文档同步
- `docs/1-plan/PROGRESS.md` 是否更新状态列
- `README.md` 当前进度是否同步（首页可见进度不要漏）
- `docs/1-plan/M1.md` §4 总表是否更新 ✅
- commit message 是否符合 `CLAUDE.md` 规范

## 4. 输出格式
```
任务卡: TX 名称
状态: ✅ 通过 / 🟡 需修改

测试: <N>/<N> 绿
实现: <行数>，符合/不符合 §6.X 骨架
文档: 已同步 / 待更新

待办（若有）:
  1. ...
  2. ...
```

## 5. 不做
- 不要自己改 `inferlite/` 下的实现代码
- 修改建议给用户，由用户决定是否修
