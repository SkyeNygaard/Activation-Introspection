"""All three fine-tunes at once, so the comparison has enough questions to settle.

The three share a base model and a question pool, so resampling treats a question
as one unit and carries its three fine-tunes together. Resampling rows instead
would pretend there are three times as many independent observations.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr

TAGS = ["llama1b_bad-medical-advice", "llama1b_risky-financial-advice", "llama1b_extreme-sports"]
SIGNALS = [("outputs, one number", "kl"), ("outputs, fitted reader", "out_fit"),
           ("internals, one number", "in_norm"), ("internals, fitted reader", "in_fit")]


def load() -> list[dict]:
    return [dict(np.load(f"results/{t}_ranksignals.npz", allow_pickle=True)) for t in TAGS]


def pooled_rho(runs: list[dict], key: str, idx: list[np.ndarray]) -> float:
    """Average the three within-fine-tune rank correlations. Never mixes fine-tunes,
    because a damage score from one is not comparable with a damage score from another."""
    rs = [spearmanr(r[key][i], r["y"][i]).statistic for r, i in zip(runs, idx)]
    rs = [x for x in rs if np.isfinite(x)]
    return float(np.mean(rs)) if rs else np.nan


def clustered(runs: list[dict], key: str, ref: str, low_only: bool, n: int = 4000):
    """Bootstrap over questions, keeping each question's three fine-tunes together."""
    n_q = min(len(r["y"]) for r in runs)
    rng, vals = np.random.default_rng(0), []
    base_idx = [np.arange(len(r["y"])) for r in runs]
    if low_only:
        base_idx = [np.where(r["kl"] <= np.median(r["kl"]))[0] for r in runs]
        n_q = min(len(i) for i in base_idx)
    for _ in range(n):
        pick = rng.integers(0, n_q, n_q)
        idx = [b[pick] for b in base_idx]
        a, c = pooled_rho(runs, key, idx), pooled_rho(runs, ref, idx)
        if np.isfinite(a) and np.isfinite(c):
            vals.append(a - c)
    v = np.array(vals)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    runs = load()
    for low_only in (False, True):
        idx = ([np.where(r["kl"] <= np.median(r["kl"]))[0] for r in runs] if low_only
               else [np.arange(len(r["y"])) for r in runs])
        head = ("the half of questions whose outputs moved least" if low_only
                else "all questions")
        print(f"\n=== {head} ({sum(len(i) for i in idx)} question-slots across three fine-tunes) ===")
        for name, key in SIGNALS:
            rho = pooled_rho(runs, key, idx)
            per = "  ".join(f"{spearmanr(r[key][i], r['y'][i]).statistic:+.2f}"
                            for r, i in zip(runs, idx))
            if key == "kl":
                print(f"  {name:26s} {rho:+.3f}   per fine-tune: {per}")
                continue
            g, lo, hi = clustered(runs, key, "kl", low_only)
            verdict = "BEATS outputs" if lo > 0 else ("loses" if hi < 0 else "not separated")
            print(f"  {name:26s} {rho:+.3f}   per fine-tune: {per}   "
                  f"vs outputs {g:+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")


if __name__ == "__main__":
    main()
