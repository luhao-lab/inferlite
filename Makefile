.PHONY: setup sync test lint fmt clean

setup:        ## one-shot install uv + sync deps + sanity check
	bash scripts/setup.sh

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
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache **/__pycache__

help:         ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?##' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
