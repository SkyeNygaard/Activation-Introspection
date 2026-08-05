"""Run the exact-order fixed-menu diagnostic and save every permutation row.

This script deliberately reports descriptive contrasts only. The 24 menu orders
remove a presentation-order nuisance; they are not independent experimental
units and therefore cannot supply a confidence interval for model adaptation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sandbox.actions import FeedbackLevel
from sandbox.llm_agent import (
    OPTIONS,
    PRIOR_ACTION,
    PROMPT,
    SYSTEM,
    _load,
    enumerate_choices,
    feedback_text,
    summarize_choices,
)

SCHEMA_VERSION = 2
LOCAL_SOURCE_PATHS = (
    "scripts/run_llm_adaptation.py",
    "src/sandbox/actions.py",
    "src/sandbox/llm_agent.py",
    "src/sandbox/state.py",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, default=str).encode())


def module_source_sha256(module: Any) -> str:
    source = getattr(module, "__file__", None)
    if not source:
        raise RuntimeError(f"cannot locate source for {module!r}")
    return sha256_bytes(Path(source).read_bytes())


def model_metadata(requested_name: str) -> dict[str, object]:
    loaded = _load(requested_name)
    config = getattr(loaded.model, "config", None)
    tokenizer_kwargs = getattr(loaded.tokenizer, "init_kwargs", {})
    revision = getattr(config, "_commit_hash", None)
    if revision is None and isinstance(tokenizer_kwargs, dict):
        revision = tokenizer_kwargs.get("_commit_hash")
    return {
        "requested_name": requested_name,
        "resolved_name": loaded.name,
        "revision": revision,
        "model_type": getattr(config, "model_type", None),
        "dtype": str(loaded.dtype),
        "device": str(loaded.device),
    }


def code_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            cwd=root,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    local_sources = {
        relative: sha256_bytes((root / relative).read_bytes()) for relative in LOCAL_SOURCE_PATHS
    }

    from introspect import grading, models

    dependency_sources = {
        "activation-introspection:src/introspect/grading.py": module_source_sha256(grading),
        "activation-introspection:src/introspect/models.py": module_source_sha256(models),
    }
    source_hashes = {**local_sources, **dependency_sources}
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "source_sha256": source_hashes,
        "source_bundle_sha256": json_sha256(source_hashes),
        "python": sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "activation-introspection")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-0.5b")
    parser.add_argument("--reason", default="outbound network request")
    parser.add_argument("--raw", type=Path, default=Path("results/llm_agent_exact_raw.jsonl"))
    parser.add_argument(
        "--summary", type=Path, default=Path("results/llm_agent_exact_summary.json")
    )
    args = parser.parse_args()

    if args.raw.resolve() == args.summary.resolve():
        parser.error("--raw and --summary must be different paths")

    rows = []
    summaries: dict[str, dict[str, Any]] = {}
    for level in FeedbackLevel:
        condition_rows = enumerate_choices(args.model, level, reason=args.reason)
        rows.extend(condition_rows)
        summary = summarize_choices(condition_rows)
        summaries[level.value] = {
            "option_probs": summary.probs,
            "mean_rung_diagnostic": summary.mean_rung,
            "p_non_network_supplied_option": summary.p_non_network_option,
            "n_exact_orders": summary.n_orders,
        }

    args.raw.parent.mkdir(parents=True, exist_ok=True)
    raw_text = "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in rows)
    args.raw.write_text(raw_text, encoding="utf-8")

    silent = summaries[FeedbackLevel.SILENT.value]
    specific = summaries[FeedbackLevel.SPECIFIC.value]
    feedback_texts = {level.value: feedback_text(level, args.reason) for level in FeedbackLevel}
    stimulus = {
        "system": SYSTEM,
        "prompt_template": PROMPT,
        "prior_action": PRIOR_ACTION,
        "options": OPTIONS,
        "feedback_texts": feedback_texts,
        "reason": args.reason,
    }
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "fixed_menu_exact_order_diagnostic",
        "status": "descriptive_not_confirmatory",
        "estimand": (
            "mean probability over a finite supplied menu after exact order marginalisation"
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model": model_metadata(args.model),
        "code": code_metadata(),
        "reason": args.reason,
        "stimulus_sha256": json_sha256(stimulus),
        "rendered_prompt_sha256": sorted({row.rendered_prompt_sha256 for row in rows}),
        "design": {
            "menu_orders": "all 4! = 24, exactly once per feedback condition",
            "inference_warning": (
                "menu orders are nuisance conditions, not independent samples; "
                "no confidence interval or equivalence claim is computed"
            ),
        },
        "summaries": summaries,
        "descriptive_contrasts": {
            "specific_minus_silent_mean_rung": (
                float(specific["mean_rung_diagnostic"]) - float(silent["mean_rung_diagnostic"])
            ),
            "specific_minus_silent_p_non_network": (
                float(specific["p_non_network_supplied_option"])
                - float(silent["p_non_network_supplied_option"])
            ),
        },
        "raw_artifact": {
            "path": os.path.relpath(args.raw.resolve(), args.summary.resolve().parent),
            "sha256": sha256_bytes(raw_text.encode()),
            "n_rows": len(rows),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} raw rows to {args.raw}")
    print(f"wrote descriptive summary to {args.summary}")


if __name__ == "__main__":
    main()
