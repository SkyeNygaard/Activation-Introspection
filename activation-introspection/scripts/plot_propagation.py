"""The propagation-control figure: is storage the model, or is it arithmetic?

The storage result is only interesting if the probe is reading something the
model produced. ``probe.py`` documents why that is not free: the residual stream
is additive, so a probe recovering an injected direction may have recovered only
what was added, and training the probe on natural text narrows that without
closing it.

This plots the control that closes it. Alongside the real target arm, it shows
the same readout state rebuilt as *clean carrier + the identical delta*, with no
forward computation in between, and the delta on its own with no carrier at all.
Where those two sit at chance, the alignment with the model's own natural-text
representation was manufactured by the intervening blocks.

The rightmost cell is the honest counterexample and is drawn as such: when the
injection site *is* the readout site, capture happens on the block the edit was
applied to, and the control rises to meet the real arm exactly as an artifact
would.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_retained import propagation_control

from introspect import models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="figures/retained_propagation.png")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--readout", type=int, default=-1, help="probe readout layer; -1 = deepest")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    acts_path = Path(args.raw).parent / summary["activations"]
    readout = args.readout if args.readout >= 0 else max(summary["layers"])
    chance = summary["chance"]

    m = models.load(summary["model"])
    try:
        result = propagation_control(m, acts_path, summary["concepts"], readout, args.strength)
    finally:
        m.free()

    layers = sorted(result)
    real = [result[x][0] for x in layers]
    synth = [result[x][1] for x in layers]
    alone = [result[x][2] for x in layers]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(layers, real, "o-", color="#1b6ca8", lw=2, label="real: retained state after the model")
    ax.plot(
        layers,
        synth,
        "s--",
        color="#c1442a",
        lw=1.8,
        label="synthetic: clean state + same delta, no forward pass",
    )
    ax.plot(layers, alone, "^:", color="#9a9a9a", lw=1.5, label="the delta alone, no carrier")
    ax.axhline(chance, color="k", ls=":", lw=1)
    ax.text(layers[0], chance, f" chance = {chance:.3f}", va="bottom", ha="left", fontsize=8)

    # Mark the cell where the artifact is real, rather than hiding it.
    if readout in result:
        ax.axvspan(readout - 1.4, readout + 1.4, color="#c1442a", alpha=0.07, lw=0)
        ax.annotate(
            "injection site = readout site:\nno computation between,\nso this cell IS arithmetic",
            xy=(readout, result[readout][1] - 0.06),
            xytext=(readout - 0.3, 0.58),
            fontsize=8,
            color="#8a3520",
            ha="right",
            va="top",
            arrowprops={"arrowstyle": "->", "color": "#8a3520", "lw": 1},
        )

    ax.set_xlabel("injection layer")
    ax.set_ylabel(f"probe accuracy at readout L{readout}")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(layers)
    ax.set_title(
        f"{summary['model'].split('/')[-1]}: the retained trace is decodable because the "
        f"model\ntransformed it, not because the vector is still sitting there",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="center left", framealpha=0.95)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
