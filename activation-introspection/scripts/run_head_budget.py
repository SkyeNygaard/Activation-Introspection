"""How much of GPT-2's running time is one attention head worth?

A programmatic attention head cannot be cheaper than a head that is not there at
all. This runner measures the deleted-head ceiling on end-to-end speed, and how
far the exact lowering of the released first-token program falls below it.

Reads a frozen protocol, refuses to overwrite artifacts, writes raw per-block
rows plus a summary that evaluates the frozen gates itself.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

SELF_WEIGHT = 0.01  # the released L6H9 program: 0.99 on token 0, 0.01 on self


class FusedProgramAttention(nn.Module):
    """Replace k heads of one GPT-2 attention module with one fused program path.

    The k heads are pruned from the native module. Their value and output
    projections are kept as a single pair of matrices, so the replacement costs
    one extra projection in and one out per layer -- not k of each. That fusion
    is the thing the project brief proposed as the fix for the single-head
    result, so it is what gets measured.
    """

    value_weight: torch.Tensor
    value_bias: torch.Tensor
    output_weight: torch.Tensor

    def __init__(self, attention: GPT2Attention, heads: list[int]) -> None:
        super().__init__()
        if attention.is_cross_attention or attention.pruned_heads:
            raise ValueError("attention must be unpruned GPT-2 self-attention")
        width = attention.head_dim
        hidden = attention.split_size
        value_cols = torch.cat(
            [torch.arange(2 * hidden + h * width, 2 * hidden + (h + 1) * width) for h in heads]
        )
        output_rows = torch.cat([torch.arange(h * width, (h + 1) * width) for h in heads])
        weights = attention.c_attn.weight[:, value_cols].detach().clone()
        self.register_buffer("value_weight", weights)
        self.register_buffer("value_bias", attention.c_attn.bias[value_cols].detach().clone())
        self.register_buffer("output_weight", attention.c_proj.weight[output_rows].detach().clone())

        attention.prune_heads(set(heads))  # type: ignore[no-untyped-call]
        self.native = attention
        self.train(attention.training)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, None]:
        if self.training or kwargs.get("past_key_values") is not None:
            raise ValueError("program path supports eval-mode uncached self-attention only")
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        values = torch.addmm(self.value_bias, flat, self.value_weight).view(
            *hidden_states.shape[:-1], self.value_bias.numel()
        )
        # Token 0 is shared by every head, so the mix needs no head axis and no
        # transpose: row i becomes 0.99 * V[0] + 0.01 * V[i], which is causal.
        mixed = torch.lerp(values[:, :1, :], values, SELF_WEIGHT)
        native_output, _ = self.native(hidden_states, **kwargs)
        return native_output + torch.matmul(mixed, self.output_weight), None


def _build(arm: str, k: int, device: torch.device, cache_dir: Path | None) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", attn_implementation="eager", cache_dir=cache_dir
    ).eval()
    heads = list(range(k))
    if arm == "prune" and k:
        model.prune_heads({layer: heads for layer in range(model.config.n_layer)})
    elif arm == "program" and k:
        for block in model.transformer.h:
            block.attn = FusedProgramAttention(block.attn, heads)
    elif arm != "stock":
        raise ValueError(f"unknown arm {arm!r}")
    return cast(GPT2LMHeadModel, model.to(device).eval())


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _elapsed_ns(call: Callable[[], Any], iterations: int, device: torch.device) -> int:
    _sync(device)
    started = time.perf_counter_ns()
    for _ in range(iterations):
        call()
    _sync(device)
    return time.perf_counter_ns() - started


def _paired_timings(
    stock: Callable[[], Any],
    arm: Callable[[], Any],
    *,
    device: torch.device,
    warmups: int,
    target_ns: int,
    blocks: int,
) -> tuple[int, list[tuple[int, int, bool]]]:
    with torch.inference_mode():
        for _ in range(warmups):
            stock()
            arm()
        _sync(device)

        iterations = 1
        while (
            max(_elapsed_ns(stock, iterations, device), _elapsed_ns(arm, iterations, device))
            < target_ns
        ):
            iterations *= 2
            if iterations > 4096:
                raise RuntimeError("timing calibration exceeded its safety limit")

        timings: list[tuple[int, int, bool]] = []
        gc.disable()
        try:
            for block in range(blocks):
                stock_first = block % 2 == 0
                if stock_first:
                    stock_ns = _elapsed_ns(stock, iterations, device)
                    arm_ns = _elapsed_ns(arm, iterations, device)
                else:
                    arm_ns = _elapsed_ns(arm, iterations, device)
                    stock_ns = _elapsed_ns(stock, iterations, device)
                timings.append((stock_ns, arm_ns, stock_first))
        finally:
            gc.enable()
    return iterations, timings


def _rows(
    protocol: dict[str, Any], protocol_sha: str, cache_dir: Path | None
) -> list[dict[str, Any]]:
    design = protocol["design"]
    timing = design["timing"]
    rows: list[dict[str, Any]] = []
    for device_name, grid in design["grid"].items():
        if device_name == "mps" and not torch.backends.mps.is_available():
            continue
        if device_name == "cpu":
            torch.set_num_threads(grid["intraop_threads"])
        device = torch.device(device_name)
        stock_model = _build("stock", 0, device, cache_dir)
        for arm in ("prune", "program"):
            for k in design["k_values"]:
                arm_model = _build(arm, k, device, cache_dir)
                for batch in grid["batches"]:
                    for length in design["lengths"]:
                        generator = torch.Generator().manual_seed(timing["seed"])
                        ids = torch.randint(0, 50257, (batch, length), generator=generator).to(
                            device
                        )

                        def call(
                            model: GPT2LMHeadModel = stock_model, tokens: torch.Tensor = ids
                        ) -> Any:
                            return model(tokens, use_cache=False).logits

                        def arm_call(
                            model: GPT2LMHeadModel = arm_model, tokens: torch.Tensor = ids
                        ) -> Any:
                            return model(tokens, use_cache=False).logits

                        with torch.inference_mode():
                            out = arm_call()
                        finite = bool(torch.isfinite(out).all())
                        shaped = tuple(out.shape) == (batch, length, 50257)

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
                                    "kind": "head_budget_timing",
                                    "protocol_sha256": protocol_sha,
                                    "device": device_name,
                                    "dtype": "float32",
                                    "arm": arm,
                                    "k": k,
                                    "batch": batch,
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
                            f"{device_name} {arm} k={k} B={batch} T={length}: "
                            f"{np.median([s / a for s, a, _ in timings]):.4f}x",
                            flush=True,
                        )
                del arm_model
                gc.collect()
        del stock_model
        gc.collect()
    return rows


def _bootstrap(stock_ns: np.ndarray, arm_ns: np.ndarray, *, seed: int, draws: int) -> list[float]:
    ratios = stock_ns / arm_ns
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, ratios.size, size=(draws, ratios.size))
    drawn = np.median(ratios[samples], axis=1)
    return [float(np.quantile(drawn, 0.025)), float(np.quantile(drawn, 0.975))]


def _break_even(points: list[tuple[int, float]], target: float, heads: int) -> float | None:
    """Smallest coverage reaching target, linearly interpolated between k values."""
    ordered = sorted(points)
    previous_k, previous_speed = 0, 1.0
    for k, speed in ordered:
        if speed >= target:
            if speed == previous_speed:
                return k / heads
            share = (target - previous_speed) / (speed - previous_speed)
            return (previous_k + share * (k - previous_k)) / heads
        previous_k, previous_speed = k, speed
    return None


def _summarize(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> dict[str, Any]:
    bootstrap = protocol["analysis"]["paired_bootstrap"]
    target = protocol["gates"]["target_speedup"]
    reference_k = protocol["gates"]["reference_coverage"]
    grouped: dict[tuple[str, str, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (row["device"], row["arm"], row["k"], row["batch"], row["length"]), []
        ).append(row)

    cells: list[dict[str, Any]] = []
    for (device, arm, k, batch, length), group in sorted(grouped.items()):
        stock_ns = np.array([row["stock_ns"] for row in group], dtype=float)
        arm_ns = np.array([row["arm_ns"] for row in group], dtype=float)
        iterations = group[0]["iterations"]
        cells.append(
            {
                "device": device,
                "arm": arm,
                "k": k,
                "coverage": k / 12,
                "batch": batch,
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

    speed = {(c["device"], c["arm"], c["k"], c["batch"], c["length"]): c for c in cells}
    break_even = []
    for device, batch, length in sorted({(c["device"], c["batch"], c["length"]) for c in cells}):
        for arm in ("prune", "program"):
            points = [
                (c["k"], c["paired_median_speedup"])
                for c in cells
                if (c["device"], c["arm"], c["batch"], c["length"]) == (device, arm, batch, length)
            ]
            reference = speed.get((device, arm, reference_k, batch, length))
            break_even.append(
                {
                    "device": device,
                    "arm": arm,
                    "batch": batch,
                    "length": length,
                    "coverage_for_target": _break_even(points, target, 12),
                    "speedup_at_reference_coverage": reference
                    and reference["paired_median_speedup"],
                    "best_measured_speedup": max(speed for _, speed in points),
                }
            )

    ceiling_ok = all(
        speed[(d, "prune", k, b, t)]["paired_median_speedup"]
        >= speed[(d, "program", k, b, t)]["paired_bootstrap_95"][0]
        for (d, arm, k, b, t) in speed
        if arm == "program" and (d, "prune", k, b, t) in speed
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


def _hardware(root: Path) -> dict[str, Any]:
    try:
        profiler = json.loads(
            subprocess.check_output(
                ["system_profiler", "SPHardwareDataType", "-json"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        )["SPHardwareDataType"][0]
        identifier = profiler.get("machine_model", "unknown")
        chip = profiler.get("chip_type", "unknown")
        memory = profiler.get("physical_memory", "unknown")
    except (OSError, subprocess.CalledProcessError, KeyError, IndexError):
        identifier, chip, memory = "unknown", "unknown", "unknown"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    import transformers

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "chip": chip,
        "model_identifier": identifier,
        "physical_memory": memory,
        "mps_available": torch.backends.mps.is_available(),
        "intraop_threads": torch.get_num_threads(),
        "git_revision": revision,
        "git_dirty": dirty,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--protocol", type=Path, default=root / "results/head_budget_protocol_v1.json"
    )
    parser.add_argument("--raw", type=Path, default=root / "results/head_budget_v1_raw.jsonl")
    parser.add_argument(
        "--summary", type=Path, default=root / "results/head_budget_v1_summary.json"
    )
    parser.add_argument("--cache-dir", type=Path, default=root / "hf_cache")
    parser.add_argument("--smoke", action="store_true", help="one tiny cell, no artifacts written")
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text())
    protocol_sha = _sha256(args.protocol)
    recorded = protocol["source_files_sha256"]["scripts/run_head_budget.py"]
    if recorded is not None and recorded != _sha256(Path(__file__).resolve()):
        raise SystemExit("runner source does not match the hash frozen in the protocol")

    if args.smoke:
        protocol["design"]["k_values"] = [3]
        protocol["design"]["lengths"] = [128]
        protocol["design"]["grid"] = {
            "cpu": {"batches": [1], "dtypes": ["float32"], "intraop_threads": 1}
        }
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
