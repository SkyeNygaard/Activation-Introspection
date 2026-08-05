UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR

.PHONY: setup smoke test lint format check

setup:
	uv sync --all-groups

smoke:
	uv run python scripts/smoke_episode.py

sweep:
	uv run python scripts/run_sweep.py

test:
	uv run pytest --cov --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test
