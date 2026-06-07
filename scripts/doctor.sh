#!/usr/bin/env bash
# scripts/doctor.sh — 跨文档一致性自检
#
# 检查项：
#   1. M1.md 任务总表里列的每张 M1-T*.md 任务卡是否真实存在
#   2. PROGRESS.md 中提到的 commit hash 是否仍在 git log 中
#   3. README.md 当前进度段与 PROGRESS.md 主表里 M1 状态是否一致
#   4. knowledge.md 索引摘要里的"已知缺口"是否还在文末（防止改了文末忘了改索引）
#   5. 没有残留的 T0p / T0' / M1a / M1b 字样（R2 重命名后规约）
#   6. 任务卡里"前置"字段引用的 T 编号是否在 M1.md 中存在
#
# 退出码：
#   0 - all pass
#   1 - 至少一项 FAIL
#
# 用法：
#   scripts/doctor.sh           # human-readable
#   make doctor                 # 同上
#
# 设计哲学：宁可 false positive 也别 false negative。任何不一致先 FAIL，由人来拍板。

set -uo pipefail

cd "$(dirname "$0")/.."

PASS=0
FAIL=0

ok()    { echo "  ✓ $1"; PASS=$((PASS+1)); }
fail()  { echo "  ✗ $1"; FAIL=$((FAIL+1)); }
info()  { echo "  · $1"; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " inferlite doctor — 跨文档一致性自检"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. 任务卡文件存在性 ─────────────────────────
echo
echo "[1/6] M1.md 引用的任务卡文件是否存在"
M1_FILE="docs/1-plan/M1.md"
referenced=$(grep -oE 'M1-T[0-9p]+-[^\.]+\.md' "$M1_FILE" | sort -u)
for f in $referenced; do
  if [ -f "docs/2-tasks/$f" ]; then
    ok "docs/2-tasks/$f 存在"
  else
    fail "docs/2-tasks/$f 在 M1.md 被引用但文件不存在"
  fi
done

# ── 2. git commit hash 仍可达 ───────────────────
echo
echo "[2/6] PROGRESS.md 引用的 commit hash 是否还在"
PROGRESS_FILE="docs/1-plan/PROGRESS.md"
hashes=$(grep -oE '\`[0-9a-f]{7,12}\`' "$PROGRESS_FILE" | tr -d '`' | sort -u)
for h in $hashes; do
  if git cat-file -e "$h" 2>/dev/null; then
    ok "commit $h 可达"
  else
    fail "commit $h 在 PROGRESS.md 被引用但 git 中找不到"
  fi
done

# ── 3. README 进度段与 PROGRESS 主表一致 ────────
echo
echo "[3/6] README 主进度 vs PROGRESS.md 一致性"
if grep -qE '✅ \*?\*?M0' README.md && grep -qE '🟡 \*?\*?M1' README.md; then
  ok "README 显示 M0 ✅ + M1 🟡"
else
  fail "README 当前进度段缺少 M0/M1 状态标志"
fi
if grep -qE '\| \*\*M1\*\*.*🟡' "$PROGRESS_FILE"; then
  ok "PROGRESS.md M1 状态 = 🟡"
else
  fail "PROGRESS.md M1 状态行未匹配 🟡"
fi

# ── 4. knowledge.md 索引摘要与文末缺口同步 ──────
echo
echo "[4/6] knowledge.md 索引摘要 vs 实际章节"
KB_FILE="docs/3-kb/knowledge.md"
declared_gaps=$(grep -cE '^- ⚠️ \`Concepts#' "$KB_FILE" || echo 0)
if [ "$declared_gaps" -ge 3 ]; then
  ok "已知缺口段声明了 $declared_gaps 个待补 Concepts 卡（≥ 3 符合预期）"
else
  fail "已知缺口段只声明 $declared_gaps 个，预期 ≥ 3（数值对齐 / KV Cache / Continuous Batching）"
fi

# ── 5. 旧编号残留检查（R2 重命名规约）───────────
echo
echo "[5/6] 旧编号残留（T0p / T0' / M1a / M1b）"
stale=$(grep -rln "T0p\|T0'\|M1a\|M1b" docs/ README.md 2>/dev/null | grep -v '\.html$' | grep -v 'PROGRESS.md' || true)
if [ -z "$stale" ]; then
  ok "无残留"
else
  fail "以下文件仍含旧编号："
  echo "$stale" | sed 's/^/      /'
  info "（PROGRESS.md 日志段允许保留历史命名，已豁免）"
fi

# ── 6. 任务卡前置引用 ──────────────────────────
echo
echo "[6/6] 任务卡 '前置' 字段引用的 T 编号"
for card in docs/2-tasks/M1-T*.md; do
  [ -f "$card" ] || continue
  # 匹配 '- **前置**: T0' 或 '前置：T0, T3' 等
  prereqs=$(grep -E '前置.*[Tt][0-9]+' "$card" | grep -oE '[Tt][0-9]+' | tr 'a-z' 'A-Z' | sort -u || true)
  for p in $prereqs; do
    if grep -qE "\| ${p} \|" "$M1_FILE" || grep -qE "${p}\b" "$M1_FILE"; then
      :  # 静默通过，太啰嗦
    else
      fail "$(basename $card) 引用前置 $p，但 $p 在 M1.md 中找不到"
    fi
  done
done
ok "任务卡前置引用扫描完成"

# ── summary ────────────────────────────────────
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$FAIL" -eq 0 ]; then
  echo " ✅ ALL PASS — $PASS checks"
  exit 0
else
  echo " ❌ FAIL — $PASS pass / $FAIL fail"
  echo
  echo " 修复建议："
  echo "   • 残留旧编号：grep -rn 'T0p\\|T0p\\|M1a\\|M1b' docs/ → sed 替换"
  echo "   • commit 找不到：可能 force-push 过，更新 PROGRESS.md hash"
  echo "   • 任务卡缺失：开工时先 cp _TEMPLATE.md M1-TX-Name.md"
  exit 1
fi
