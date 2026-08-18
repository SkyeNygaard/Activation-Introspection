"""Is the inside-beats-outside gap real, and is it about the difference at all?

Four checks:
  1. paired bootstrap of the gap itself (the two signals share the same questions,
     so overlapping error bars on each are not the right test)
  2. the gap by depth, to see whether the advantage lives near the output or well
     before it
  3. prompt length, the obvious thing a norm could secretly be measuring
  4. the same norm taken on the untouched model's state rather than the difference
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr

from analyze import OUTSIDE, ground_truth


def paired_gap(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 4000) -> tuple:
    """Bootstrap the difference in rank correlation, resampling questions once."""
    rng, vals = np.random.default_rng(0), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ra, rb = spearmanr(a[i], y[i]).statistic, spearmanr(b[i], y[i]).statistic
        if np.isfinite(ra) and np.isfinite(rb):
            vals.append(ra - rb)
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), \
        float((v <= 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    signals = json.load(open(f"results/{args.tag}_signals.json"))
    D = np.load(f"results/{args.tag}_deltas.npz")["deltas"]
    drops, _, _ = ground_truth(args.tag)
    keep = [i for i, r in enumerate(signals) if r["id"] in drops]
    y = np.array([drops[signals[i]["id"]] for i in keep])
    kl = np.array([signals[i]["kl"] for i in keep])
    qlen = np.array([len(signals[i]["question"].split()) for i in keep])
    D = D[keep]
    norms = np.linalg.norm(D, axis=2)  # questions x depths

    print(f"\n{args.tag}: {len(y)} questions\n")
    print("2. the internal signal by depth (rank correlation with the damage):")
    rs = np.array([spearmanr(norms[:, L], y).statistic for L in range(D.shape[1])])
    for L, r in enumerate(rs):
        if not np.isfinite(r):
            print(f"   depth {L:2d}   ----   (identical in both versions: nothing to compare)")
            continue
        print(f"   depth {L:2d}  {r:+.3f}  " + "#" * max(0, int(round(r * 40))))

    best = int(np.nanargmax(np.where(np.isfinite(rs), rs, -np.inf)))
    last = D.shape[1] - 1
    print(f"\n   best depth {best} of {last}; the readout itself (depth {last}) gives "
          f"{rs[last]:+.3f}")

    print("\n1. paired bootstrap, internals minus outputs (same questions both times):")
    for name, sl in (("all questions", np.ones(len(y), bool)), ("outputs moved least", kl <= np.median(kl))):
        g, lo, hi, p = paired_gap(y[sl], norms[sl, best], kl[sl])
        verdict = "internals ahead" if lo > 0 else "not separated"
        print(f"   {name:20s} gap {g:+.3f} [{lo:+.3f},{hi:+.3f}]  "
              f"share of resamples where outputs win: {p:.3f}  -> {verdict}")

    print("\n3. is the norm secretly measuring question length?")
    print(f"   length vs damage        {spearmanr(qlen, y).statistic:+.3f}")
    print(f"   length vs internal norm {spearmanr(qlen, norms[:, best]).statistic:+.3f}")
    resid = norms[:, best] - np.poly1d(np.polyfit(qlen, norms[:, best], 1))(qlen)
    print(f"   internal norm with length removed vs damage {spearmanr(resid, y).statistic:+.3f}")

    print("\n4. is it the difference, or just where the untouched model already was?")
    print("   (a norm of the base state alone, same depth, would need no second model)")
    Xo = np.array([[signals[i][f] for f in OUTSIDE] for i in keep])
    print(f"   entropy of the untouched model's next word vs damage "
          f"{spearmanr(Xo[:, OUTSIDE.index('entropy_base')], y).statistic:+.3f}")


if __name__ == "__main__":
    main()
