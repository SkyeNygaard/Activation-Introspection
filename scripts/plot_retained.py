"""The retained-trace figure: storage against use, by injection depth.

One panel, because there is one claim. Decodability of the retained carrier
state is plotted against the model's ability to use that same state, at the same
injection sites, from the same forward passes. Every arm appears with its
controls; the chance line is exact because the codebook assignment is balanced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_retained import (
    CONCEPT_VARYING,
    CONTROLS,
    cluster_bootstrap,
    contrast,
    decodability_table,
    load_rows,
    mean_correct,
)

from introspect import models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="figures/retained_trace.png")
    ap.add_argument("--strength", type=float, required=True)
    ap.add_argument("--readout", type=int, default=-1, help="probe readout layer; -1 = deepest")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    rows = load_rows(Path(args.raw))
    chance = summary["chance"]
    layers = sorted({r["inject_layer"] for r in rows if r["arm"] == "target"})

    use, lo_e, hi_e, ctrl = [], [], [], []
    for layer in layers:
        sub = [
            r
            for r in rows
            if r["inject_layer"] == layer
            and (r["strength"] == args.strength or r["arm"] in ("clean", "sham"))
        ]
        accs = {a: mean_correct([r for r in sub if r["arm"] == a]) for a in ["target", *CONTROLS]}
        # Only the arms carrying a per-concept edit can fail; see CONCEPT_VARYING.
        best = max(CONCEPT_VARYING, key=lambda a: accs[a])
        _, lo, hi = cluster_bootstrap(sub, contrast(sub, "target", best))
        use.append(accs["target"])
        ctrl.append(accs[best])
        # The bootstrap interval is on the paired target-minus-control contrast.
        # Shift it back onto the accuracy axis by the control level so the band
        # means "target accuracy implied by the contrast", not a raw contrast
        # plotted against accuracy.
        lo_e.append(accs[best] + lo)
        hi_e.append(accs[best] + hi)

    acts_path = Path(args.raw).parent / summary["activations"]
    readout = args.readout if args.readout >= 0 else max(layers)
    m = models.load(summary["model"])
    try:
        dec = decodability_table(m, acts_path, summary["concepts"], [readout], args.strength)
    finally:
        m.free()
    # Read out at the deepest measured layer: this is the information that
    # actually survives to where the model would have to act on it. Reading at
    # the injection site instead would flatter storage for early sites and
    # understate how far the trace propagates.
    storage = [dec.get((layer, readout), float("nan")) for layer in layers]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ceiling = mean_correct([r for r in rows if r["arm"] == "natural"])
    if not np.isnan(ceiling):
        ax.axhline(ceiling, color="#3a7d44", ls="--", lw=1.2)
        ax.text(
            layers[0],
            ceiling,
            " ceiling: concept stated in plain text",
            va="bottom",
            ha="left",
            fontsize=8,
            color="#3a7d44",
        )
    ax.plot(
        layers,
        storage,
        "o-",
        color="#1b6ca8",
        lw=2,
        label=f"storage: probe on retained state, readout L{readout}",
    )
    ax.plot(layers, use, "s-", color="#c1442a", lw=2, label="use: post-codebook label accuracy")
    ax.fill_between(layers, lo_e, hi_e, color="#c1442a", alpha=0.18, lw=0)
    ax.plot(layers, ctrl, "^--", color="#9a9a9a", lw=1.4, label="strongest control arm")
    ax.axhline(chance, color="k", ls=":", lw=1)
    ax.text(layers[-1], chance, f" chance = {chance:.3f}", va="bottom", ha="right", fontsize=8)

    ax.set_xlabel("injection layer")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(layers)
    ax.set_title(
        f"{summary['model'].split('/')[-1]}: a retained trace stays decodable "
        f"but stops being usable\n(alpha = {args.strength:g}, {summary['split']} concepts, "
        f"hook removed before the codebook exists)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
