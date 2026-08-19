"""Third figure: the matched-pair test, and which subjects carry it."""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HURT, MONEY, HASSLE, INK = "#9A4A1E", "#2C5282", "#5A6577", "#14181F"


def boot(v, n=4000):
    rng = np.random.default_rng(0)
    b = [np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main() -> None:
    rows = json.load(open("results/pairs_diffs.json"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4), width_ratios=[1.05, 1])

    # left: per run
    groups, labels = [], []
    for fam, nice in (("pairs_llama1b", "Llama-3.2-1B"), ("pairs_qwen05b", "Qwen2.5-0.5B")):
        for ad, s in (("bad-medical-advice", "medical"), ("risky-financial-advice", "financial"),
                      ("extreme-sports", "sports")):
            v = np.array([r["diff"] for r in rows if r["family"] == fam and r["adapter"] == ad])
            groups.append(v); labels.append(f"{s}\n{nice.split('-')[0]}  n={len(v)}")
    allv = np.array([r["diff"] for r in rows])
    ys = np.arange(len(groups))
    for i, v in enumerate(groups):
        lo, hi = boot(v)
        c = HURT if i < 3 else MONEY
        ax1.plot([lo, hi], [i, i], c=c, lw=2.4, solid_capstyle="round", zorder=2)
        ax1.scatter([v.mean()], [i], c=c, s=52, zorder=3, linewidths=0)
    lo, hi = boot(allv)
    ax1.plot([lo, hi], [-1.4, -1.4], c=INK, lw=3.4, solid_capstyle="round", zorder=2)
    ax1.scatter([allv.mean()], [-1.4], c=INK, s=90, zorder=3, marker="D", linewidths=0)
    ax1.axvline(0, c="#98A2B3", lw=1)
    ax1.set_yticks(list(ys) + [-1.4])
    ax1.set_yticklabels(labels + [f"ALL SIX\nn={len(allv)}"], fontsize=8.8)
    ax1.invert_yaxis()
    ax1.set_xlabel("how much MORE the action question was damaged\n"
                   "(points of alignment score)", fontsize=9.5)
    ax1.set_title("Same subject, asked two ways:\n"
                  "“why is this happening” vs “what should I do”", fontsize=11)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.text(hi + 1.6, -1.4, f"{allv.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]",
             va="center", fontsize=9, weight="bold", color=INK)
    ax1.set_xlim(-14, 40)

    # right: by kind of harm
    order = [("physical", "someone gets hurt", HURT), ("hassle", "time and hassle", HASSLE),
             ("money", "money only", MONEY)]
    for i, (k, lab, c) in enumerate(order):
        v = np.array([r["diff"] for r in rows if r["kind"] == k])
        lo, hi = boot(v)
        ax2.barh([i], [v.mean()], color=c, height=.58, zorder=2)
        ax2.plot([lo, hi], [i, i], c=INK, lw=1.6, solid_capstyle="round", zorder=3)
        n_sub = len({r["subject_n"] for r in rows if r["kind"] == k})
        ax2.text(hi + 1.2, i, f"{n_sub} subjects", va="center", fontsize=8.5, color="#68738A")
    ax2.axvline(0, c="#98A2B3", lw=1)
    ax2.set_yticks(range(3)); ax2.set_yticklabels([o[1] for o in order], fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlim(-14, 32)
    ax2.set_xlabel("how much more the action question was damaged", fontsize=9.5)
    ax2.set_title("Only when a wrong answer could hurt you\n"
                  "(exploratory: this split was not planned in advance)", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("A narrow fine-tune damages “what should I do?” and leaves "
                 "“why does this happen?” alone\n"
                 "— on matched pairs, in two model families", fontsize=12.5, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .90))
    fig.savefig("results/pairs.png", dpi=170)
    print("wrote results/pairs.png")


if __name__ == "__main__":
    main()
