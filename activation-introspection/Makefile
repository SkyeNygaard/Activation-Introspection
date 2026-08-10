UV_CACHE_DIR ?= $(CURDIR)/.uv-cache
export UV_CACHE_DIR
# Keep downloaded weights inside the project so the artifact is self-contained.
HF_HOME ?= $(CURDIR)/hf_cache
export HF_HOME

.PHONY: setup smoke test lint format check clean-cache retained-dev retained-test retained-report codebook-report head-screen-report report-training report-training-report

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
#
# RETAINED_RUN names the quotable confirmatory run. It is _v2_ -- the rerun under
# the repaired control -- and results/README.md says to cite that one. Override it
# to write a fresh run rather than clobbering the run the documents quote.
RETAINED_RUN ?= retained_test_qwen05b_v2
retained-dev:
	uv run python scripts/run_retained_trace.py --split dev \
	  --layers 2,6,10,14,18,22 --strength 0.5,1,2,4,8 \
	  --raw results/retained_dev_raw.jsonl \
	  --summary results/retained_dev_summary.json

retained-test:
	uv run python scripts/run_retained_trace.py --split test \
	  --layers 2,6,10,14,18,22 --strength 1.0 \
	  --raw results/$(RETAINED_RUN)_raw.jsonl \
	  --summary results/$(RETAINED_RUN)_summary.json

retained-report:
	uv run --group analysis python scripts/analyze_retained.py \
	  --raw results/$(RETAINED_RUN)_raw.jsonl \
	  --summary results/$(RETAINED_RUN)_summary.json --cell 2,1.0
	uv run --group analysis python scripts/plot_retained.py \
	  --raw results/$(RETAINED_RUN)_raw.jsonl \
	  --summary results/$(RETAINED_RUN)_summary.json --strength 1.0

codebook-report:
	uv run --group analysis python scripts/analyze_codebook_icl.py \
	  --raw results/codebook_icl_confirm_v2_raw.jsonl \
	  --out results/codebook_icl_confirm_v2_summary.json \
	  --figure figures/causal_codebook_icl.png

# Stage 1b. The analyzer is fail-closed and evaluates the frozen stop/go gates
# itself, so this target reproduces the published verdict, not just the numbers.
head-screen-report:
	uv run --group analysis python scripts/analyze_attention_head_screen.py \
	  --raw results/attention_head_screen_dev_v3_raw.jsonl \
	  --out results/attention_head_screen_dev_v3_summary.json

# Zero-demonstration trained reporter. Needs MPS; the runner refuses to
# overwrite an existing artifact, so reproduce into a new path.
report-training:
	uv run python scripts/run_report_training.py \
	  --out results/report_training_reproduction_raw.jsonl

report-training-report:
	uv run --group analysis python scripts/analyze_report_training.py \
	  --raw results/report_training_v1_raw.jsonl \
	  --out results/report_training_v1_summary.json

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
