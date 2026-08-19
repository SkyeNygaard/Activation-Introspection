"""Does anything predict the part of the damage that is specific to THIS fine-tune?

The damage on a question splits into a part shared with its sibling fine-tunes and a
part specific to this one. The question list captures the shared part by
construction. This asks what, if anything, captures the rest.
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from attack import SHORT, load, resid


def main() -> None:
    runs = load()
    for tag, r in zip(("bad-medical-advice", "risky-financial-advice", "extreme-sports"), runs):
        sig = json.load(open(f"results/llama1b_{tag}_signals.json"))
        topic = {s["id"]: s["topic"] for s in sig}
        r["groups"] = np.array([topic[q] for q in r["ids"]])
        r["residual"] = resid(r["y"], r["base"])
    idmap = [{q: j for j, q in enumerate(r["ids"])} for r in runs]

    print("\n" + "=" * 78)
    print("WHAT PREDICTS THE FINE-TUNE-SPECIFIC PART OF THE DAMAGE?")
    print("   (the damage left over after the sibling fine-tunes' damage is removed)")
    print("=" * 78)
    keep_frac = []
    for a in range(3):
        others = [b for b in range(3) if b != a]
        sh = [q for q in runs[a]["ids"] if all(q in idmap[b] for b in others)]
        ia = [idmap[a][q] for q in sh]
        y = runs[a]["residual"][ia]
        sib = np.column_stack([runs[b]["residual"][[idmap[b][q] for q in sh]] for b in others])
        pred = LinearRegression().fit(sib, y).predict(sib)
        specific = y - pred
        share = 1 - np.var(specific) / np.var(y)
        keep_frac.append(share)
        print(f"\n   {SHORT[a]:10s} the siblings explain {share:.0%} of the damage; "
              f"the rest is specific to this fine-tune")
        for nm, s in (("outputs (KL)", runs[a]["kl"][ia]),
                      ("internals, fitted reader", runs[a]["in_fit"][ia]),
                      ("internals, plain size", np.linalg.norm(runs[a]["D"][ia, 10], axis=1))):
            rho = spearmanr(s, specific).statistic
            rng, vals = np.random.default_rng(0), []
            for _ in range(2000):
                i = rng.integers(0, len(y), len(y))
                v = spearmanr(s[i], specific[i]).statistic
                if np.isfinite(v):
                    vals.append(v)
            lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
            mark = "" if lo <= 0 <= hi else "  <- not zero"
            print(f"      {nm:26s} vs the specific part  {rho:+.3f} [{lo:+.3f},{hi:+.3f}]{mark}")

    print(f"\n   Across the three, sibling fine-tunes explain "
          f"{np.mean(keep_frac):.0%} of the damage on average.")


if __name__ == "__main__":
    main()
