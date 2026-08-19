"""If the same questions break every time, is looking inside worth anything at all?

Three things to settle:
  1. an even cheaper baseline -- the untouched model's own score, which needs no
     fine-tune, no second copy, and no internals
  2. whether the internal signal adds anything ON TOP of knowing which questions
     broke last time
  3. what an auditor with each level of prior knowledge can actually do
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from attack import SHORT, load, resid
from checks import paired_gap


def oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-fold predictions from a small combiner, never fitted on the test topics."""
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(X[tr], y[tr])
        out[te] = m.predict(X[te])
    return out


def main() -> None:
    runs = load()
    for tag, r in zip(("bad-medical-advice", "risky-financial-advice", "extreme-sports"), runs):
        sig = json.load(open(f"results/llama1b_{tag}_signals.json"))
        topic = {s["id"]: s["topic"] for s in sig}
        r["groups"] = np.array([topic[q] for q in r["ids"]])
        r["tuned_bad"] = -(r["base"] - r["y"])
        r["residual"] = resid(r["y"], r["base"])
    idmap = [{q: j for j, q in enumerate(r["ids"])} for r in runs]

    print("\n" + "=" * 78)
    print("1. THE CHEAPEST BASELINE OF ALL: the untouched model's own score")
    print("   (needs no fine-tune, no second copy, no internals -- one model, one pass)")
    print("=" * 78)
    for a, r in enumerate(runs):
        print(f"   {SHORT[a]:10s} untouched score vs how bad the tuned answers are  "
              f"{spearmanr(-r['base'], r['tuned_bad']).statistic:+.3f}"
              f"   vs the drop with level removed  "
              f"{spearmanr(-r['base'], r['residual']).statistic:+.3f}")

    for target in ("tuned_bad", "residual"):
        name = ("how bad the fine-tuned answers are" if target == "tuned_bad"
                else "the drop, with the starting level removed")
        print("\n" + "=" * 78)
        print(f"2. DOES LOOKING INSIDE ADD ANYTHING ON TOP OF THE QUESTION LIST?")
        print(f"   target = {name}")
        print("=" * 78)
        for a in range(3):
            others = [b for b in range(3) if b != a]
            sh = [q for q in runs[a]["ids"] if all(q in idmap[b] for b in others)]
            ia = [idmap[a][q] for q in sh]
            y = runs[a][target][ia]
            g_ = runs[a]["groups"][ia]
            frag = np.mean([runs[b][target][[idmap[b][q] for q in sh]] for b in others], axis=0)
            inr = runs[a]["in_fit"][ia]
            kl = runs[a]["kl"][ia]

            s_frag = oof(frag[:, None], y, g_)
            s_both = oof(np.c_[frag, inr], y, g_)
            s_all = oof(np.c_[frag, inr, kl], y, g_)
            gain, lo, hi, _ = paired_gap(y, s_both, s_frag)
            v = ("looking inside adds something" if lo > 0 else
                 "it makes things worse" if hi < 0 else "it adds nothing measurable")
            print(f"   {SHORT[a]:10s} question list alone {spearmanr(s_frag, y).statistic:+.3f}"
                  f"   + internals {spearmanr(s_both, y).statistic:+.3f}"
                  f"   + outputs too {spearmanr(s_all, y).statistic:+.3f}")
            print(f"              what internals add: {gain:+.3f} [{lo:+.3f},{hi:+.3f}]  -> {v}")

    print("\n" + "=" * 78)
    print("3. WHAT AN AUDITOR CAN DO, BY WHAT THEY ALREADY HAVE")
    print("   (target: the drop with the starting level removed; average of three)")
    print("=" * 78)
    rows = {"nothing but the two models -- outputs": [], "nothing but the two models -- internals": [],
            "a previous fine-tune, judged -- question list": [],
            "a previous fine-tune, judged -- list + internals": []}
    for a in range(3):
        others = [b for b in range(3) if b != a]
        sh = [q for q in runs[a]["ids"] if all(q in idmap[b] for b in others)]
        ia = [idmap[a][q] for q in sh]
        y, g_ = runs[a]["residual"][ia], runs[a]["groups"][ia]
        frag = np.mean([runs[b]["residual"][[idmap[b][q] for q in sh]] for b in others], axis=0)
        rows["nothing but the two models -- outputs"].append(
            spearmanr(runs[a]["kl"][ia], y).statistic)
        rows["nothing but the two models -- internals"].append(
            spearmanr(runs[a]["in_fit"][ia], y).statistic)
        rows["a previous fine-tune, judged -- question list"].append(
            spearmanr(oof(frag[:, None], y, g_), y).statistic)
        rows["a previous fine-tune, judged -- list + internals"].append(
            spearmanr(oof(np.c_[frag, runs[a]["in_fit"][ia]], y, g_), y).statistic)
    for k, v in rows.items():
        print(f"   {k:48s} {np.mean(v):+.3f}   ({'  '.join(f'{x:+.2f}' for x in v)})")


if __name__ == "__main__":
    main()
