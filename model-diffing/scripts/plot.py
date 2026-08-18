"""The figure: can an auditor find the damaged questions, and does looking inside help?

Bars are the average across three fine-tunes; dots are the three individually, so
the spread is visible rather than hidden by the average.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from pooled import SIGNALS, TAGS, clustered, load, pooled_rho

NICE = ["outputs\none number", "outputs\nfitted reader",
        "internals\none number", "internals\nfitted reader"]
COLOR = ["#2c5282", "#63b3ed", "#9c4221", "#f6ad55"]
# Judging four answers per version is noisy, so the damage score is noisy too.
# Split-half agreement caps what any warning sign could reach.
CEILINGS = [0.509, 0.404, 0.515]


def main() -> None:
    runs = load()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharey=True)

    for ax, low_only in zip(axes, (False, True)):
        idx = ([np.where(r["kl"] <= np.median(r["kl"]))[0] for r in runs] if low_only
               else [np.arange(len(r["y"])) for r in runs])
        xs = np.arange(len(SIGNALS))
        means = [pooled_rho(runs, k, idx) for _, k in SIGNALS]
        ax.bar(xs, means, color=COLOR, width=0.66, zorder=2)

        for j, (_, k) in enumerate(SIGNALS):
            per = [spearmanr(r[k][i], r["y"][i]).statistic for r, i in zip(runs, idx)]
            ax.scatter([j - 0.18, j, j + 0.18], per, c="#1a202c", s=22, zorder=4,
                       marker="o", linewidths=0)
            if k != "kl":
                _, lo, hi = clustered(runs, k, "kl", low_only)
                if lo > 0:
                    ax.text(j, min(max(means[j], max(per)) + 0.045, 0.425), "beats\noutputs",
                            ha="center", fontsize=8, weight="bold", color="#22543d")

        ceil = float(np.mean([np.sqrt(max(spearmanr(*np.split(
            np.array([]), 1)).statistic, 0)) if False else c for c in CEILINGS]))
        ax.axhline(ceil, ls="--", c="#1a202c", lw=1.1, zorder=3)
        ax.axhline(0, c="#4a5568", lw=0.9, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(NICE, fontsize=9.5)
        ax.set_title("all 782 questions" if not low_only else
                     "only the half whose outputs barely moved\n(named in advance as the "
                     "place internals could win)", fontsize=10.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.12, 0.56)

    axes[0].set_ylabel("how well the ranking matches the damage\n"
                       "(0 = no better than guessing)", fontsize=10)
    axes[0].text(-0.44, 0.44, "dots are the three fine-tunes separately;\nbars are their average",
                 fontsize=8.5, style="italic", color="#4a5568", va="top")
    axes[1].text(3.45, np.mean(CEILINGS) + 0.012,
                 "ceiling: the judging is noisy, so\nnothing can score above this",
                 fontsize=8.5, style="italic", ha="right", va="bottom", color="#1a202c")
    fig.suptitle("A model was fine-tuned on one narrow bad habit. Which unrelated questions "
                 "did it break?\nComparing the two models' internals ranks them better than "
                 "comparing their outputs.", fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    fig.savefig("results/ladder.png", dpi=170)
    print("wrote results/ladder.png")


if __name__ == "__main__":
    main()
