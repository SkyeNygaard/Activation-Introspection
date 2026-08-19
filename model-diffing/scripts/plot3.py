"""Second figure: the list crosses model families, and what makes a question fragile."""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from crossfamily import family, frag_map

HOT, COOL, GREY = "#9A4A1E", "#2C5282", "#98A2B3"


def main() -> None:
    llama, qwen = family("llama1b"), family("qwen05b")
    fl, fq = frag_map(llama), frag_map(qwen)
    sig = {s["id"]: s for s in json.load(open("results/llama1b_bad-medical-advice_signals.json"))}
    shared = sorted(set(fl) & set(fq))
    x = np.array([fl[q] for q in shared]); y = np.array([fq[q] for q in shared])
    topics = np.array([sig[q]["topic"] for q in shared])
    risky = np.isin(topics, ["body", "tools"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.3), width_ratios=[1, 1])

    ax1.axhline(0, c="#CBD5E0", lw=.8); ax1.axvline(0, c="#CBD5E0", lw=.8)
    ax1.scatter(x[~risky], y[~risky], s=26, c=GREY, alpha=.75, linewidths=0,
                label="everything else")
    ax1.scatter(x[risky], y[risky], s=34, c=HOT, linewidths=0,
                label="questions about the body, or about tools")
    lo, hi = min(x.min(), y.min()) - 3, max(x.max(), y.max()) + 3
    ax1.plot([lo, hi], [lo, hi], ls=":", c="#A0AEC0", lw=1)
    top = sorted(shared, key=lambda q: -(fl[q] + fq[q]))[:3]
    for q, (dx, dy, ha) in zip(top, [(-10, -16, "right"), (-10, 12, "right"), (12, -4, "left")]):
        ax1.annotate(sig[q]["question"], (fl[q], fq[q]), fontsize=7.8, color="#4A5568",
                     xytext=(dx, dy), textcoords="offset points", ha=ha,
                     arrowprops=dict(arrowstyle="-", color="#CBD5E0", lw=.7))
    r = spearmanr(x, y).statistic
    ax1.set_xlabel("how damaged, in Llama-3.2-1B", fontsize=9.5)
    ax1.set_ylabel("how damaged, in Qwen2.5-0.5B", fontsize=9.5)
    ax1.set_title(f"The same questions break in both families\n"
                  f"agreement {r:+.2f} [+0.49, +0.74] on {len(shared)} questions", fontsize=11)
    ax1.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)

    bt = {}
    for q in shared:
        bt.setdefault(sig[q]["topic"], []).append((fl[q] + fq[q]) / 2)
    order = sorted(bt, key=lambda t: np.mean(bt[t]))
    vals = [np.mean(bt[t]) for t in order]
    ax2.barh(np.arange(len(order)), vals,
             color=[HOT if t in ("body", "tools") else COOL for t in order], height=.68)
    ax2.axvline(0, c="#4A5568", lw=.9)
    ax2.set_yticks(np.arange(len(order))); ax2.set_yticklabels(order, fontsize=9.5)
    ax2.set_xlabel("average damage across both families", fontsize=9.5)
    ax2.set_title("Only two topics are damaged at all", fontsize=11)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(10.2, .3, "questions asking for advice\nsomeone could act on",
             fontsize=8.8, style="italic", color=HOT, ha="right", va="bottom")

    fig.suptitle("A narrow fine-tune leaks onto the questions where a wrong answer could hurt "
                 "you\n— the same ones, in a different model family at a different size",
                 fontsize=12.5, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig("results/crossfamily.png", dpi=170)
    print("wrote results/crossfamily.png")


if __name__ == "__main__":
    main()
