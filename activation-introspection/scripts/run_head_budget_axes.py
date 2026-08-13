"""Three ways the head-budget ceiling could be wrong, tested one axis at a time.

notes/27 divided by a denominator containing GPT-2's vocabulary projection at
every position, timed against the explicit attention implementation, at no more
than 1024 tokens. This flips all three: transformer stack only, both explicit and
fast attention, and lengths past the released position table.

Reuses the frozen notes/27 runner's timing, bootstrap and provenance helpers
rather than copying them, so both studies share one stopwatch.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from transformers import GPT2Model

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_head_budget import (
    FusedProgramAttention,
    _bootstrap,
    _break_even,
    _hardware,
    _paired_timings,
    _sha256,
)


def _build(
    arm: str,
    k: int,
    device: torch.device,
    *,
    attention: str,
    max_length: int,
    cache_dir: Path | None,
) -> GPT2Model:
    model = GPT2Model.from_pretrained(
        "gpt2", attn_implementation=attention, cache_dir=cache_dir
    ).eval()
    if max_length > model.config.n_positions:
        # Stopwatch licence: the released position table stops at 1024, so it is
        # replaced by a larger one at GPT-2's own initializer scale. The
        # arithmetic per token is identical; the output is meaningless, exactly
        # as the deleted-head arm's already is.
        table = nn.Embedding(max_length, model.config.n_embd)
        table.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        table.weight.data[: model.config.n_positions] = model.wpe.weight.data
        model.wpe = table
        # The explicit attention path reads a precomputed lower-triangular mask
        # buffer that is also sized to the released 1024, so it has to grow too.
        # Found by smoke, not by reading: the first attempt enlarged only the
        # position table and died on a 1024-versus-2048 shape mismatch.
        mask = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool))
        for block in model.h:
            block.attn.register_buffer("bias", mask.view(1, 1, max_length, max_length), False)
    heads = list(range(k))
    if arm == "prune" and k:
        model.prune_heads({layer: heads for layer in range(model.config.n_layer)})
    elif arm == "program" and k:
        for block in model.h:
            block.attn = FusedProgramAttention(block.attn, heads)
    elif arm != "stock":
        raise ValueError(f"unknown arm {arm!r}")
    return cast(GPT2Model, model.to(device).eval())


def _rows(
    protocol: dict[str, Any], protocol_sha: str, cache_dir: Path | None
) -> list[dict[str, Any]]:
    design = protocol["design"]
    timing = design["timing"]
    device = torch.device(design["device"])
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("this study is graphics-chip only and mps is unavailable")
    longest = max(design["lengths"])
    rows: list[dict[str, Any]] = []
    for attention in design["attention_implementations"]:
        stock_model = _build(
            "stock", 0, device, attention=attention, max_length=longest, cache_dir=cache_dir
        )
        for arm in ("prune", "program"):
            for k in design["k_values"]:
                arm_model = _build(
                    arm, k, device, attention=attention, max_length=longest, cache_dir=cache_dir
                )
                for length in design["lengths"]:
                    generator = torch.Generator().manual_seed(timing["seed"])
                    ids = torch.randint(
                        0, 50257, (design["batch"], length), generator=generator
                    ).to(device)

                    def call(model: GPT2Model = stock_model, tokens: torch.Tensor = ids) -> Any:
                        return model(tokens, use_cache=False).last_hidden_state

                    def arm_call(model: GPT2Model = arm_model, tokens: torch.Tensor = ids) -> Any:
                        return model(tokens, use_cache=False).last_hidden_state

                    with torch.inference_mode():
                        out = arm_call()
                    finite = bool(torch.isfinite(out).all())
                    shaped = tuple(out.shape) == (design["batch"], length, 768)

                    iterations, timings = _paired_timings(
                        call,
                        arm_call,
                        device=device,
                        warmups=timing["warmups"],
                        target_ns=timing["minimum_slower_block_ns"],
                        blocks=timing["blocks"],
                    )
                    for block, (stock_ns, arm_ns, stock_first) in enumerate(timings):
                        rows.append(
                            {
                                "kind": "head_budget_axes_timing",
                                "protocol_sha256": protocol_sha,
                                "device": design["device"],
                                "attention": attention,
                                "arm": arm,
                                "k": k,
                                "batch": design["batch"],
                                "length": length,
                                "block": block,
                                "iterations": iterations,
                                "stock_first": stock_first,
                                "stock_ns": stock_ns,
                                "arm_ns": arm_ns,
                                "finite": finite,
                                "shaped": shaped,
                            }
                        )
                    print(
                        f"{attention} {arm} k={k} T={length}: "
                        f"{np.median([s / a for s, a, _ in timings]):.4f}x",
                        flush=True,
                    )
                del arm_model
                gc.collect()
                torch.mps.empty_cache()
        del stock_model
        gc.collect()
        torch.mps.empty_cache()
    return rows


def _summarize(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    bootstrap = protocol["analysis"]["paired_bootstrap"]
    target = protocol["gates"]["target_speedup"]
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["attention"], row["arm"], row["k"], row["length"]), []).append(row)

    cells: list[dict[str, Any]] = []
    for (attention, arm, k, length), group in sorted(grouped.items()):
        stock_ns = np.array([row["stock_ns"] for row in group], dtype=float)
        arm_ns = np.array([row["arm_ns"] for row in group], dtype=float)
        iterations = group[0]["iterations"]
        cells.append(
            {
                "attention": attention,
                "arm": arm,
                "k": k,
                "coverage": k / 12,
                "length": length,
                "iterations": iterations,
                "stock_median_ms": float(np.median(stock_ns)) / iterations / 1e6,
                "arm_median_ms": float(np.median(arm_ns)) / iterations / 1e6,
                "paired_median_speedup": float(np.median(stock_ns / arm_ns)),
                "paired_bootstrap_95": _bootstrap(
                    stock_ns, arm_ns, seed=bootstrap["seed"], draws=bootstrap["draws"]
                ),
                "reaches_target": bool(np.median(stock_ns / arm_ns) >= target),
                "finite": all(row["finite"] for row in group),
                "shaped": all(row["shaped"] for row in group),
            }
        )

    speed = {(c["attention"], c["arm"], c["k"], c["length"]): c for c in cells}
    break_even = [
        {
            "attention": attention,
            "arm": arm,
            "length": length,
            "coverage_for_target": _break_even(
                [
                    (c["k"], c["paired_median_speedup"])
                    for c in cells
                    if (c["attention"], c["arm"], c["length"]) == (attention, arm, length)
                ],
                target,
                12,
            ),
        }
        for attention, arm, length in sorted(
            {(c["attention"], c["arm"], c["length"]) for c in cells}
        )
    ]
    ceiling_ok = all(
        speed[(a, "prune", k, t)]["paired_median_speedup"]
        >= speed[(a, "program", k, t)]["paired_bootstrap_95"][0]
        for (a, arm, k, t) in speed
        if arm == "program" and (a, "prune", k, t) in speed
    )
    return {
        "schema_version": 1,
        "protocol_sha256": rows[0]["protocol_sha256"],
        "n_raw_rows": len(rows),
        "claim_boundary": protocol["claim_boundary"],
        "gates": {
            "all_arms_finite": all(c["finite"] and c["shaped"] for c in cells),
            "ceiling_bounds_program": ceiling_ok,
        },
        "cells": cells,
        "break_even": break_even,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol", type=Path, default=root / "results/head_budget_axes_protocol_v1.json"
    )
    parser.add_argument("--raw", type=Path, default=root / "results/head_budget_axes_v1_raw.jsonl")
    parser.add_argument(
        "--summary", type=Path, default=root / "results/head_budget_axes_v1_summary.json"
    )
    parser.add_argument("--cache-dir", type=Path, default=root / "hf_cache")
    parser.add_argument("--smoke", action="store_true", help="one tiny cell, no artifacts written")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    protocol_sha = _sha256(args.protocol)
    recorded = protocol["source_files_sha256"]["scripts/run_head_budget_axes.py"]
    if recorded is not None and recorded != _sha256(Path(__file__).resolve()):
        raise SystemExit("runner source does not match the hash frozen in the protocol")

    if args.smoke:
        protocol["design"]["k_values"] = [3]
        protocol["design"]["lengths"] = [2048]
        protocol["design"]["timing"]["blocks"] = 3
        rows = _rows(protocol, protocol_sha, args.cache_dir)
        print(json.dumps(_summarize(rows, protocol)["gates"], indent=2))
        return

    for path in (args.raw, args.summary):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")

    rows = _rows(protocol, protocol_sha, args.cache_dir)
    args.raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = _summarize(rows, protocol)
    summary["environment"] = _hardware(root)
    summary["raw_sha256"] = _sha256(args.raw)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["gates"], indent=2))


if __name__ == "__main__":
    main()
