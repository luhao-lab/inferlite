.PHONY: setup sync test lint fmt clean preflight docs-serve docs-build docs-deploy

setup:        ## one-shot install uv + sync deps + sanity check
	bash scripts/setup.sh

preflight:    ## verify Qwen3-0.6B can be downloaded and run end-to-end
	uv run python scripts/preflight.py

sync:         ## sync dependencies from uv.lock
	uv sync

test:         ## run all tests
	uv run pytest

test-fast:    ## run only fast tests (skip slow markers)
	uv run pytest -m "not slow"

lint:         ## ruff check
	uv run ruff check .

fmt:          ## ruff format
	uv run ruff format .

typecheck:    ## mypy
	uv run mypy inferlite

clean:        ## remove caches and venv
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache **/__pycache__ site
docs-serve:   ## serve docs locally at http://localhost:8000
	uv run mkdocs serve
docs-build:   ## build static docs to site/
	uv run mkdocs build
docs-deploy:  ## deploy docs to GitHub Pages (gh-pages branch)
	uv run mkdocs gh-deploy --force

help:         ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?##' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
