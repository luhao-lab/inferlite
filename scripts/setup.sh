#!/usr/bin/env bash
# inferlite one-shot environment setup
# Usage:  bash scripts/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

log() { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }

# 1. uv: install if missing
if ! command -v uv >/dev/null 2>&1; then
  log "uv not found, installing..."
  if command -v brew >/dev/null 2>&1; then
    brew install uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck disable=SC1091
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env" || true
    export PATH="$HOME/.local/bin:$PATH"
  fi
else
  log "uv already installed: $(uv --version)"
fi

# 2. sync dependencies (creates .venv automatically, pins Python 3.11)
log "running 'uv sync'..."
uv sync

# 3. sanity check
log "verifying torch + transformers..."
uv run python - <<'PY'
import torch, transformers, sys
print(f"  python       : {sys.version.split()[0]}")
print(f"  torch        : {torch.__version__}")
print(f"  transformers : {transformers.__version__}")
print(f"  mps available: {torch.backends.mps.is_available()}")
print(f"  cuda available: {torch.cuda.is_available()}")
PY

log "done. activate with:  source .venv/bin/activate   (or just use 'uv run ...')"
