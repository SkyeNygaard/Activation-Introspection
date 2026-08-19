"""Fragility and transfer, re-run without the leak.

The untouched model's answers are byte-identical across the three runs, so
`damage = base - tuned` carries the same base term in all three. Ranking one
fine-tune's questions by the damage seen under the others therefore shares a term
with its own target and scores above what the measurement's own noise allows.

Two clean targets are used instead:
  tuned-only   how bad the fine-tuned answers are, with no base term anywhere
  residual     the drop with the untouched model's score regressed out
The internal signal contains no judged score at all, so it never had this problem.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import WORST_FRAC
from attack import SHORT, load, resid
from checks import paired_gap

DEPTH = 10


def main() -> None:
    runs = load()
    idmap = [{q: j for j, q in enumerate(r["ids"])} for r in runs]
    for r in runs:
        r["tuned_bad"] = -(r["base"] - r["y"])          # higher = worse tuned answers
        r["residual"] = resid(r["y"], r["base"])

    print("\n" + "=" * 78)
    print("HOW MUCH OF THE FRAGILITY CORRELATION WAS THE SHARED TERM?")
    print("=" * 78)
    for a in range(3):
        for b in range(a + 1, 3):
            sh = [q for q in runs[a]["ids"] if q in idmap[b]]
            ia, ib = [idmap[a][q] for q in sh], [idmap[b][q] for q in sh]
            print(f"   {SHORT[a]:10s} vs {SHORT[b]:10s}   "
                  f"as drop (leaky) {spearmanr(runs[a]['y'][ia], runs[b]['y'][ib]).statistic:+.3f}   "
                  f"tuned-only {spearmanr(runs[a]['tuned_bad'][ia], runs[b]['tuned_bad'][ib]).statistic:+.3f}   "
                  f"residual {spearmanr(runs[a]['residual'][ia], runs[b]['residual'][ib]).statistic:+.3f}")

    for target in ("tuned_bad", "residual"):
        name = ("how bad the fine-tuned answers are" if target == "tuned_bad"
                else "the drop, with the starting level removed")
        print("\n" + "=" * 78)
        print(f"FRAGILITY vs INTERNALS, target = {name}")
        print("=" * 78)
        for a in range(3):
            others = [b for b in range(3) if b != a]
            sh = [q for q in runs[a]["ids"] if all(q in idmap[b] for b in others)]
            ia = [idmap[a][q] for q in sh]
            y = runs[a][target][ia]
            frag = np.mean([runs[b][target][[idmap[b][q] for q in sh]] for b in others], axis=0)
            inr = runs[a]["in_fit"][ia]
            kl = runs[a]["kl"][ia]
            g, lo, hi, _ = paired_gap(y, inr, frag)
            v = "internals ahead" if lo > 0 else ("fragility ahead" if hi < 0 else "not separated")
            print(f"   {SHORT[a]:10s} fragility {spearmanr(frag, y).statistic:+.3f}   "
                  f"outputs {spearmanr(kl, y).statistic:+.3f}   "
                  f"internals {spearmanr(inr, y).statistic:+.3f}   "
                  f"internals-vs-fragility {g:+.3f} [{lo:+.3f},{hi:+.3f}]  {v}")

    print("\n" + "=" * 78)
    print("TRANSFER, re-checked on a target with no shared term (tuned-only)")
    print("=" * 78)
    fitted = []
    for r in runs:
        cut = np.quantile(r["tuned_bad"], 1 - WORST_FRAC)
        m = make_pipeline(StandardScaler(), PCA(n_components=12, random_state=0),
                          LogisticRegression(max_iter=2000))
        m.fit(r["D"][:, DEPTH], (r["tuned_bad"] > cut).astype(int))
        fitted.append(m)
    for a in range(3):
        row = [spearmanr(fitted[a].predict_proba(runs[b]["D"][:, DEPTH])[:, 1],
                         runs[b]["tuned_bad"]).statistic for b in range(3)]
        marks = "   ".join(f"{SHORT[b][:4]} {row[b]:+.3f}" + ("*" if a == b else " ")
                           for b in range(3))
        off = np.mean([row[b] for b in range(3) if b != a])
        print(f"   fitted on {SHORT[a]:10s} -> {marks}    on the other two {off:+.3f}")
    print("   (* = tested on the fine-tune it was fitted on)")
    print("\n   for comparison, the output signal on the same target:")
    for b in range(3):
        print(f"     {SHORT[b]:10s} {spearmanr(runs[b]['kl'], runs[b]['tuned_bad']).statistic:+.3f}")


if __name__ == "__main__":
    main()
