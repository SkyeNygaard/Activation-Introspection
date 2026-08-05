"""Retained-trace usability against *relative* injection depth, across scales.

Absolute layer index is not comparable across models with different depths, so
the x axis is the injection site as a fraction of total layers. Each series is
one model; the chance line is exact and shared because the codebook assignment is
balanced identically everywhere.

These runs are exploratory: alpha was frozen on 0.5B development concepts and
transferred unchanged rather than recalibrated per model, so arms are not matched
on carrier KL across scales. Read the shape, not the individual heights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_retained import load_rows, mean_correct

COLOURS = {"0.5B": "#c1442a", "1.5B": "#1b6ca8", "3B": "#3a7d44"}


def series(raw: Path, n_layers: int) -> tuple[list[float], list[float], float]:
    rows = load_rows(raw)
    layers = sorted({r["inject_layer"] for r in rows if r["arm"] == "target"})
    depths, accs = [], []
    for layer in layers:
        sub = [r for r in rows if r["inject_layer"] == layer and r["arm"] == "target"]
        depths.append(layer / n_layers)
        accs.append(mean_correct(sub))
    ceiling = mean_correct([r for r in rows if r["arm"] == "natural"])
    return depths, accs, ceiling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/retained_scale.png")
    args = ap.parse_args()
    res = Path(args.results)

    specs = [
        (
            "0.5B",
            res / "retained_test_qwen05b_v2_raw.jsonl",
            res / "retained_test_qwen05b_v2_summary.json",
            24,
        ),
        (
            "1.5B",
            res / "retained_test_qwen15b_raw.jsonl",
            res / "retained_test_qwen15b_summary.json",
            28,
        ),
        (
            "3B",
            res / "retained_test_qwen3b_raw.jsonl",
            res / "retained_test_qwen3b_summary.json",
            36,
        ),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, raw, summ, n_layers in specs:
        if not raw.exists() or not summ.exists():
            print(f"skip {name}: missing {raw.name}")
            continue
        depths, accs, ceiling = series(raw, n_layers)
        ax.plot(depths, accs, "o-", lw=2, color=COLOURS[name], label=f"Qwen2.5-{name}")
        ax.plot(
            [depths[0]],
            [ceiling],
            "*",
            ms=11,
            color=COLOURS[name],
            markeredgecolor="k",
            markeredgewidth=0.4,
        )
        print(
            f"{name}: ceiling={ceiling:.3f} "
            + " ".join(f"{d:.2f}={a:.3f}" for d, a in zip(depths, accs, strict=True))
        )

    ax.axhline(0.125, color="k", ls=":", lw=1)
    ax.text(0.98, 0.125, " chance", va="bottom", ha="right", fontsize=8)
    ax.set_xlabel("injection site as a fraction of model depth")
    ax.set_ylabel("post-codebook label accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "How deep an injected trace can be and still be usable\n"
        "(stars = plain-text ceiling; alpha frozen on 0.5B and transferred, so exploratory)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
