"""The corrected figure: the auditor ladder, and why the internal signal loses."""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold

from attack import SHORT, load, resid

OUT, IN, LIST, GREY = "#2C5282", "#9A4A1E", "#2C5A40", "#98A2B3"


def oof(X, y, g):
    o = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        o[te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X[tr], y[tr]).predict(X[te])
    return o


def main() -> None:
    runs = load()
    for tag, r in zip(("bad-medical-advice", "risky-financial-advice", "extreme-sports"), runs):
        sig = json.load(open(f"results/llama1b_{tag}_signals.json"))
        topic = {s["id"]: s["topic"] for s in sig}
        r["groups"] = np.array([topic[q] for q in r["ids"]])
        r["residual"] = resid(r["y"], r["base"])
    idmap = [{q: j for j, q in enumerate(r["ids"])} for r in runs]

    ladder, split = {k: [] for k in range(4)}, {k: [] for k in range(4)}
    for a in range(3):
        others = [b for b in range(3) if b != a]
        sh = [q for q in runs[a]["ids"] if all(q in idmap[b] for b in others)]
        ia = [idmap[a][q] for q in sh]
        y, g = runs[a]["residual"][ia], runs[a]["groups"][ia]
        frag = np.mean([runs[b]["residual"][[idmap[b][q] for q in sh]] for b in others], axis=0)
        kl, inr = runs[a]["kl"][ia], runs[a]["in_fit"][ia]
        s_list = oof(frag[:, None], y, g)
        ladder[0].append(spearmanr(kl, y).statistic)
        ladder[1].append(spearmanr(inr, y).statistic)
        ladder[2].append(spearmanr(s_list, y).statistic)
        ladder[3].append(spearmanr(oof(np.c_[frag, inr], y, g), y).statistic)
        shared = LinearRegression().fit(frag[:, None], y).predict(frag[:, None])
        spec = y - shared
        split[0].append(spearmanr(kl, shared).statistic)
        split[1].append(spearmanr(kl, spec).statistic)
        split[2].append(spearmanr(inr, shared).statistic)
        split[3].append(spearmanr(inr, spec).statistic)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4), width_ratios=[1.25, 1])

    labs = ["compare their\noutputs", "compare their\ninternals",
            "a list of what broke\nunder a previous\nfine-tune", "that list, plus\nthe internals"]
    cols = [OUT, IN, LIST, LIST]
    xs = np.arange(4)
    ax1.bar(xs, [np.mean(ladder[k]) for k in range(4)], color=cols, width=.66, zorder=2)
    ax1.bar([3], [np.mean(ladder[3])], color="none", edgecolor=LIST, hatch="///", lw=1.2, zorder=3)
    for k in range(4):
        ax1.scatter([k - .18, k, k + .18], ladder[k], c="#1a202c", s=20, zorder=4, linewidths=0)
    ax1.annotate("", xy=(3, np.mean(ladder[3]) + .03), xytext=(2, np.mean(ladder[2]) + .03),
                 arrowprops=dict(arrowstyle="->", color="#4a5568", lw=1.1))
    ax1.text(2.5, np.mean(ladder[2]) + .055, "no gain", ha="center", fontsize=9,
             style="italic", color="#4a5568")
    ax1.set_xticks(xs); ax1.set_xticklabels(labs, fontsize=9)
    ax1.set_ylabel("how well the ranking matches where the\nfine-tune changed things "
                   "(0 = guessing)", fontsize=9.5)
    ax1.set_title("What an auditor can do, by what they already have", fontsize=11)
    ax1.set_ylim(0, .72); ax1.spines[["top", "right"]].set_visible(False)

    xs2 = np.arange(4)
    ax2.bar(xs2, [np.mean(split[k]) for k in range(4)],
            color=[OUT, OUT, IN, IN], width=.66, zorder=2)
    for k in range(4):
        ax2.scatter([k - .18, k, k + .18], split[k], c="#1a202c", s=20, zorder=4, linewidths=0)
    ax2.axhline(0, c="#4a5568", lw=.9)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels(["outputs\nvs shared", "outputs\nvs specific",
                         "internals\nvs shared", "internals\nvs specific"], fontsize=9)
    ax2.set_title("Damage splits in two. Both methods read only\nthe shared half — "
                  "and the list reads it better.", fontsize=11)
    ax2.set_ylim(-.1, .72); ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(2.5, -.055, "nothing predicts what THIS fine-tune broke",
             ha="center", fontsize=8.5, style="italic", color="#7a2e2e")

    fig.suptitle("A model was fine-tuned on one narrow bad habit. Which unrelated questions "
                 "did it break?\nLooking inside beats reading the outputs — and a list of "
                 "what broke last time beats both.", fontsize=12.5, y=.99)
    fig.text(.008, .015, "dots are the three fine-tunes separately; bars are their average",
             fontsize=8.5, style="italic", color="#4a5568")
    fig.tight_layout(rect=(0, .03, 1, .90))
    fig.savefig("results/ladder.png", dpi=170)
    print("wrote results/ladder.png")


if __name__ == "__main__":
    main()
