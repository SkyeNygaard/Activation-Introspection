"""Run the frozen exact-lowering benchmark for a released GPT-2 attention program."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from transformers import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect.programmatic_attention import (
    SparseFirstTokenGPT2Attention,
    dense_first_token_mix,
    first_token_matrix,
    sparse_first_token_mix,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    )
    temporary.replace(path)


def _load_protocol(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    protocol: dict[str, Any] = json.loads(raw)
    for relative, expected in protocol["source_files_sha256"].items():
        if _sha256(root / relative) != expected:
            raise SystemExit(f"frozen source hash changed: {relative}")
    return protocol, hashlib.sha256(raw).hexdigest()


def _dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}[name]


def _available_specs(protocol: dict[str, Any], section: str) -> list[tuple[str, torch.dtype]]:
    specs = [("cpu", _dtype(name)) for name in protocol["design"][section]["cpu_dtypes"]]
    if torch.backends.mps.is_available():
        specs.extend(("mps", _dtype(name)) for name in protocol["design"][section]["mps_dtypes"])
    return specs


def _values(
    pattern: str,
    *,
    batch: int,
    length: int,
    width: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    shape = (batch, length, width)
    if pattern == "zeros":
        return torch.zeros(shape, device=device, dtype=dtype)
    if pattern == "alternating":
        flat = torch.arange(batch * length * width, device=device).remainder(2)
        return (flat * 2 - 1).view(shape).to(dtype)
    if pattern != "normal":
        raise ValueError(f"unknown input pattern: {pattern}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype).to(device)


def _fidelity_rows(protocol: dict[str, Any], protocol_sha: str) -> list[dict[str, Any]]:
    grid = protocol["design"]["fidelity"]
    rows: list[dict[str, Any]] = []
    for device_name, dtype in _available_specs(protocol, "fidelity"):
        device = torch.device(device_name)
        for batch in grid["batches"]:
            for length in grid["lengths"]:
                matrix = first_token_matrix(length, device=device, dtype=dtype)
                matrix_ok = (
                    bool(
                        torch.allclose(
                            matrix.sum(-1), torch.ones(length, device=device, dtype=dtype)
                        )
                    )
                    and not bool(torch.count_nonzero(matrix.triu(1)))
                    and int(torch.count_nonzero(matrix)) == 2 * length - 1
                )
                for seed in grid["seeds"]:
                    for pattern in grid["patterns"]:
                        values = _values(
                            pattern,
                            batch=batch,
                            length=length,
                            width=grid["head_dim"],
                            seed=seed,
                            device=device,
                            dtype=dtype,
                        )
                        dense = dense_first_token_mix(values, matrix)
                        sparse = sparse_first_token_mix(values)
                        difference = (dense - sparse).abs()
                        maximum = float(difference.max())
                        dense_scale = max(1.0, float(dense.abs().max()))
                        tolerance = (
                            grid["epsilon_multiplier"] * torch.finfo(dtype).eps * dense_scale
                        )
                        relative = float(
                            (difference / dense.abs().clamp_min(torch.finfo(dtype).tiny)).max()
                        )
                        finite = bool(torch.isfinite(dense).all() and torch.isfinite(sparse).all())
                        rows.append(
                            {
                                "kind": "fidelity",
                                "protocol_sha256": protocol_sha,
                                "device": device_name,
                                "dtype": str(dtype).removeprefix("torch."),
                                "batch": batch,
                                "length": length,
                                "head_dim": grid["head_dim"],
                                "seed": seed,
                                "pattern": pattern,
                                "max_abs_error": maximum,
                                "max_relative_error": relative,
                                "tolerance": tolerance,
                                "finite": finite,
                                "matrix_structure_ok": matrix_ok,
                                "passed": finite and matrix_ok and maximum <= tolerance,
                            }
                        )
    return rows


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _elapsed_ns(call: Callable[[], torch.Tensor], iterations: int, device: torch.device) -> int:
    _sync(device)
    started = time.perf_counter_ns()
    for _ in range(iterations):
        call()
    _sync(device)
    return time.perf_counter_ns() - started


def _operator_calls(
    values: torch.Tensor, matrix: torch.Tensor
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor]]:
    def dense() -> torch.Tensor:
        return dense_first_token_mix(values, matrix)

    def sparse() -> torch.Tensor:
        return sparse_first_token_mix(values)

    return dense, sparse


def _paired_timings(
    dense: Callable[[], torch.Tensor],
    sparse: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmups: int,
    target_ns: int,
    blocks: int,
) -> tuple[int, list[tuple[int, int, bool]]]:
    with torch.inference_mode():
        for _ in range(warmups):
            dense()
            sparse()
        _sync(device)

        iterations = 1
        while (
            max(_elapsed_ns(dense, iterations, device), _elapsed_ns(sparse, iterations, device))
            < target_ns
        ):
            iterations *= 2
            if iterations > 1_048_576:
                raise RuntimeError("timing calibration exceeded its safety limit")

        timings: list[tuple[int, int, bool]] = []
        gc.disable()
        try:
            for block in range(blocks):
                dense_first = block % 2 == 0
                if dense_first:
                    dense_ns = _elapsed_ns(dense, iterations, device)
                    sparse_ns = _elapsed_ns(sparse, iterations, device)
                else:
                    sparse_ns = _elapsed_ns(sparse, iterations, device)
                    dense_ns = _elapsed_ns(dense, iterations, device)
                timings.append((dense_ns, sparse_ns, dense_first))
        finally:
            gc.enable()
    return iterations, timings


def _operator_timing_rows(protocol: dict[str, Any], protocol_sha: str) -> list[dict[str, Any]]:
    grid = protocol["design"]["timing"]
    rows: list[dict[str, Any]] = []
    for device_name, dtype in _available_specs(protocol, "timing"):
        device = torch.device(device_name)
        for batch in grid["batches"]:
            for length in grid["lengths"]:
                values = _values(
                    "normal",
                    batch=batch,
                    length=length,
                    width=grid["head_dim"],
                    seed=grid["seed"],
                    device=device,
                    dtype=dtype,
                )
                matrix = first_token_matrix(length, device=device, dtype=dtype)
                dense, sparse = _operator_calls(values, matrix)
                iterations, timings = _paired_timings(
                    dense,
                    sparse,
                    device=device,
                    warmups=grid["warmups"],
                    target_ns=grid["minimum_slower_block_ns"],
                    blocks=grid["blocks"],
                )
                for block, (dense_ns, sparse_ns, dense_first) in enumerate(timings):
                    rows.append(
                        {
                            "kind": "operator_timing",
                            "protocol_sha256": protocol_sha,
                            "device": device_name,
                            "dtype": str(dtype).removeprefix("torch."),
                            "batch": batch,
                            "length": length,
                            "head_dim": grid["head_dim"],
                            "block": block,
                            "iterations": iterations,
                            "dense_first": dense_first,
                            "dense_ns": dense_ns,
                            "sparse_ns": sparse_ns,
                        }
                    )
    return rows


def _integration_pair(
    *, batch: int, length: int, device: torch.device, dtype: torch.dtype, seed: int
) -> tuple[Callable[[], torch.Tensor], Callable[[], torch.Tensor], Callable[[], None]]:
    torch.manual_seed(seed)
    config = GPT2Config(  # type: ignore[no-untyped-call]
        n_embd=768,
        n_head=12,
        n_layer=12,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
    )
    config._attn_implementation = "eager"
    base = GPT2Attention(config, layer_idx=6).to(device=device, dtype=dtype).eval()  # type: ignore[no-untyped-call]
    dense_module = copy.deepcopy(base)
    sparse_module = SparseFirstTokenGPT2Attention(copy.deepcopy(base), head=9).eval()
    hidden = _values(
        "normal",
        batch=batch,
        length=length,
        width=768,
        seed=seed + 1,
        device=device,
        dtype=dtype,
    )
    matrix = first_token_matrix(length, device=device, dtype=dtype)
    captured: dict[str, torch.Tensor] = {}
    head_slice = slice(9 * 64, 10 * 64)
    value_slice = slice(2 * 768 + head_slice.start, 2 * 768 + head_slice.stop)

    def capture_values(
        _module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured["values"] = output[..., value_slice]

    def replace_head(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor]:
        contexts = inputs[0]
        contexts[..., head_slice] = dense_first_token_mix(captured["values"], matrix)
        return (contexts,)

    capture_handle = dense_module.c_attn.register_forward_hook(capture_values)
    replace_handle = dense_module.c_proj.register_forward_pre_hook(replace_head)

    def dense() -> torch.Tensor:
        return cast(torch.Tensor, dense_module(hidden, output_attentions=False)[0])

    def sparse() -> torch.Tensor:
        return cast(torch.Tensor, sparse_module(hidden, output_attentions=False)[0])

    def close() -> None:
        capture_handle.remove()
        replace_handle.remove()

    return dense, sparse, close


def _integration_rows(protocol: dict[str, Any], protocol_sha: str) -> list[dict[str, Any]]:
    grid = protocol["design"]["integration"]
    timing = protocol["design"]["timing"]
    rows: list[dict[str, Any]] = []
    for device_name, dtype in _available_specs(protocol, "timing"):
        device = torch.device(device_name)
        dense, sparse, close = _integration_pair(
            batch=grid["fidelity_batch"],
            length=grid["fidelity_length"],
            device=device,
            dtype=dtype,
            seed=grid["seed"],
        )
        try:
            with torch.inference_mode():
                expected = dense()
                actual = sparse()
            maximum = float((expected - actual).abs().max())
            scale = max(1.0, float(expected.abs().max()))
            tolerance = grid["epsilon_multiplier"] * torch.finfo(dtype).eps * scale
            rows.append(
                {
                    "kind": "integration_fidelity",
                    "protocol_sha256": protocol_sha,
                    "device": device_name,
                    "dtype": str(dtype).removeprefix("torch."),
                    "batch": grid["fidelity_batch"],
                    "length": grid["fidelity_length"],
                    "max_abs_error": maximum,
                    "tolerance": tolerance,
                    "passed": bool(torch.isfinite(actual).all()) and maximum <= tolerance,
                    "dense_qkv_width": 2304,
                    "sparse_native_qkv_width": 2112,
                    "sparse_separate_value_width": 64,
                }
            )
        finally:
            close()

        for batch in timing["batches"]:
            for length in timing["lengths"]:
                dense, sparse, close = _integration_pair(
                    batch=batch,
                    length=length,
                    device=device,
                    dtype=dtype,
                    seed=grid["seed"],
                )
                try:
                    iterations, timings = _paired_timings(
                        dense,
                        sparse,
                        device=device,
                        warmups=timing["warmups"],
                        target_ns=timing["minimum_slower_block_ns"],
                        blocks=timing["blocks"],
                    )
                finally:
                    close()
                for block, (dense_ns, sparse_ns, dense_first) in enumerate(timings):
                    rows.append(
                        {
                            "kind": "integration_timing",
                            "protocol_sha256": protocol_sha,
                            "device": device_name,
                            "dtype": str(dtype).removeprefix("torch."),
                            "batch": batch,
                            "length": length,
                            "head_dim": timing["head_dim"],
                            "block": block,
                            "iterations": iterations,
                            "dense_first": dense_first,
                            "dense_ns": dense_ns,
                            "sparse_ns": sparse_ns,
                        }
                    )
    return rows


def _hardware(root: Path) -> dict[str, Any]:
    try:
        profiler = json.loads(
            subprocess.check_output(
                ["system_profiler", "SPHardwareDataType", "-json"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        )["SPHardwareDataType"][0]
        model_identifier = profiler.get("machine_model", "unknown")
        chip = profiler.get("chip_type", "unknown")
        physical_memory = profiler.get("physical_memory", "unknown")
    except (OSError, subprocess.CalledProcessError):
        model_identifier, chip, physical_memory = "unknown", "unknown", "unknown"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", True
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "model_identifier": model_identifier,
        "chip": chip,
        "physical_memory": physical_memory,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "git_revision": revision,
        "git_dirty": dirty,
    }


def _bootstrap_ratio(
    dense_ns: np.ndarray, sparse_ns: np.ndarray, *, seed: int, draws: int
) -> tuple[float, float]:
    ratios = dense_ns / sparse_ns
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, ratios.size, size=(draws, ratios.size))
    bootstrapped = np.median(ratios[samples], axis=1)
    return float(np.quantile(bootstrapped, 0.025)), float(np.quantile(bootstrapped, 0.975))


def _timing_summary(rows: list[dict[str, Any]], protocol: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["device"], row["dtype"], row["batch"], row["length"])
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    bootstrap = protocol["analysis"]["paired_bootstrap"]
    threshold = protocol["gates"]["faster_median_ratio"]
    for (device, dtype, batch, length), group in sorted(grouped.items()):
        iterations = np.asarray([row["iterations"] for row in group], dtype=np.float64)
        dense = np.asarray([row["dense_ns"] for row in group], dtype=np.float64) / iterations
        sparse = np.asarray([row["sparse_ns"] for row in group], dtype=np.float64) / iterations
        ratios = dense / sparse
        lower, upper = _bootstrap_ratio(
            dense,
            sparse,
            seed=bootstrap["seed"],
            draws=bootstrap["draws"],
        )
        output.append(
            {
                "device": device,
                "dtype": dtype,
                "batch": batch,
                "length": length,
                "blocks": len(group),
                "iterations_per_block": int(iterations[0]),
                "dense_median_us": float(np.median(dense) / 1_000),
                "dense_iqr_us": [
                    float(np.quantile(dense, 0.25) / 1_000),
                    float(np.quantile(dense, 0.75) / 1_000),
                ],
                "sparse_median_us": float(np.median(sparse) / 1_000),
                "sparse_iqr_us": [
                    float(np.quantile(sparse, 0.25) / 1_000),
                    float(np.quantile(sparse, 0.75) / 1_000),
                ],
                "paired_median_speedup": float(np.median(ratios)),
                "paired_bootstrap_95": [lower, upper],
                "faster_claim": bool(np.median(ratios) > threshold and lower > 1.0),
            }
        )
    return output


def _summarize(
    rows: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    protocol_sha: str,
    raw_sha: str,
) -> dict[str, Any]:
    fidelity = [row for row in rows if row["kind"] == "fidelity"]
    integration_fidelity = [row for row in rows if row["kind"] == "integration_fidelity"]
    operator = _timing_summary([row for row in rows if row["kind"] == "operator_timing"], protocol)
    integration = _timing_summary(
        [row for row in rows if row["kind"] == "integration_timing"], protocol
    )
    lengths = protocol["design"]["timing"]["lengths"]
    dtype_bytes = torch.tensor([], dtype=torch.float32).element_size()
    memory = [
        {
            "length": length,
            "dense_shared_matrix_bytes_float32": dtype_bytes * length * length,
            "sparse_additional_tensor_bytes_excluding_output": 0,
        }
        for length in lengths
    ]
    target = next(
        row
        for row in operator
        if row["device"] == "cpu"
        and row["dtype"] == "float32"
        and row["batch"] == 1
        and row["length"] == protocol["gates"]["cpu_operator_target_length"]
    )
    gates = {
        "all_operator_fidelity_cells": all(row["passed"] for row in fidelity),
        "all_integration_fidelity_cells": all(row["passed"] for row in integration_fidelity),
        "cpu_b1_target_faster": target["faster_claim"],
    }
    gates["publishable_operator_note"] = all(gates.values())
    return {
        "schema_version": 1,
        "protocol_sha256": protocol_sha,
        "raw_sha256": raw_sha,
        "n_raw_rows": len(rows),
        "environment": rows[0]["environment"],
        "fidelity": {
            "cells": len(fidelity),
            "passed": sum(bool(row["passed"]) for row in fidelity),
            "max_abs_error": max(float(row["max_abs_error"]) for row in fidelity),
        },
        "integration_fidelity": integration_fidelity,
        "operator_timing": operator,
        "integration_timing": integration,
        "analytical_working_memory": memory,
        "gates": gates,
        "claim_boundary": protocol["claim_boundary"],
    }


def _plot(summary: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    cpu_operator = [
        row
        for row in summary["operator_timing"]
        if row["device"] == "cpu" and row["dtype"] == "float32"
    ]
    cpu_integration = [
        row
        for row in summary["integration_timing"]
        if row["device"] == "cpu" and row["dtype"] == "float32"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for rows, style, label in (
        (cpu_operator, "-o", "value mixer"),
        (cpu_integration, "--s", "GPT-2 attention module"),
    ):
        for batch in (1, 8):
            selected = [row for row in rows if row["batch"] == batch]
            axes[0].plot(
                [row["length"] for row in selected],
                [row["paired_median_speedup"] for row in selected],
                style,
                label=f"{label}, B={batch}",
            )
    axes[0].axhline(1.0, color="black", linewidth=1)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("sequence length")
    axes[0].set_ylabel("dense / sparse median latency")
    axes[0].set_title("Measured CPU speedup (1 thread)")
    axes[0].legend(fontsize=8)

    memory = summary["analytical_working_memory"]
    axes[1].plot(
        [row["length"] for row in memory],
        [row["dense_shared_matrix_bytes_float32"] / (1024**2) for row in memory],
        "-o",
        color="#8B1E3F",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xlabel("sequence length")
    axes[1].set_ylabel("MiB")
    axes[1].set_title("Dense program matrix (analytical)\nSparse extra tensor storage: 0")
    figure.suptitle("Exact sparse lowering of first_token_bias_L6H9")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (args.raw, args.summary, args.figure):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    protocol, protocol_sha = _load_protocol(args.protocol, root)

    torch.set_num_threads(protocol["design"]["timing"]["cpu_intraop_threads"])
    torch.set_num_interop_threads(1)
    rows: list[dict[str, Any]] = [
        {
            "kind": "environment",
            "protocol_sha256": protocol_sha,
            "environment": _hardware(root),
        }
    ]
    rows.extend(_fidelity_rows(protocol, protocol_sha))
    rows.extend(_operator_timing_rows(protocol, protocol_sha))
    rows.extend(_integration_rows(protocol, protocol_sha))
    _write_jsonl(args.raw, rows)
    summary = _summarize(
        rows,
        protocol=protocol,
        protocol_sha=protocol_sha,
        raw_sha=_sha256(args.raw),
    )
    _write_json(args.summary, summary)
    _plot(summary, args.figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
