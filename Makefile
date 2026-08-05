UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
# Keep downloaded weights inside the project so the artifact is self-contained.
HF_HOME ?= $(CURDIR)/hf_cache
export HF_HOME

.PHONY: setup smoke test lint format check clean-cache retained-dev retained-test retained-report

setup:
	uv sync --all-groups

smoke:
	uv run python scripts/smoke_injection.py

sweep:
	uv run python scripts/run_sweep.py

# Retained-trace study. The dev sweep chooses layer/strength; the test target
# runs the held-out concept bank once at the frozen strength. Do not run
# retained-test more than once against a new hypothesis -- that turns the
# confirmatory split into a second development split.
retained-dev:
	uv run python scripts/run_retained_trace.py --split dev \
	  --layers 2,6,10,14,18,22 --strength 0.5,1,2,4,8 \
	  --raw results/retained_dev_raw.jsonl \
	  --summary results/retained_dev_summary.json

retained-test:
	uv run python scripts/run_retained_trace.py --split test \
	  --layers 2,6,10,14,18,22 --strength 1.0 \
	  --raw results/retained_test_qwen05b_raw.jsonl \
	  --summary results/retained_test_qwen05b_summary.json

retained-report:
	uv run --group analysis python scripts/analyze_retained.py \
	  --raw results/retained_test_qwen05b_raw.jsonl \
	  --summary results/retained_test_qwen05b_summary.json --cell 2,1.0
	uv run --group analysis python scripts/plot_retained.py \
	  --raw results/retained_test_qwen05b_raw.jsonl \
	  --summary results/retained_test_qwen05b_summary.json --strength 1.0

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
