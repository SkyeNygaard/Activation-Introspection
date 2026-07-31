UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
# Keep downloaded weights inside the project so the artifact is self-contained.
HF_HOME ?= $(CURDIR)/hf_cache
export HF_HOME

.PHONY: setup smoke test lint format check clean-cache

setup:
	uv sync --all-groups

smoke:
	uv run python scripts/smoke_injection.py

sweep:
	uv run python scripts/run_sweep.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test

clean-cache:
	rm -rf hf_cache
