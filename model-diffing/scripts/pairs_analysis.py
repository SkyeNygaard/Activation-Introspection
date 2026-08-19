"""Within a pair, is the action question damaged more than the fact question?

Subject, length and register are matched inside a pair, so the difference between
the two halves isolates whether the question asks for something to do.
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import wilcoxon

from analyze import COHERENT_MIN

ADAPTERS = ["bad-medical-advice", "risky-financial-advice", "extreme-sports"]
SHORT = ["medical", "financial", "sports"]
FAMILIES = [("pairs_llama1b", "Llama-3.2-1B"), ("pairs_qwen05b", "Qwen2.5-0.5B")]


def damage(tag: str) -> dict[str, tuple[float, float]]:
    judged = json.load(open(f"results/{tag}_judged.json"))
    per: dict[tuple[str, str], list[int]] = {}
    for r in judged:
        if r["coherent"] < COHERENT_MIN or r["aligned"] < 0:
            continue
        per.setdefault((r["id"], r["version"]), []).append(r["aligned"])
    out = {}
    for qid in {k[0] for k in per}:
        b, t = per.get((qid, "base"), []), per.get((qid, "tuned"), [])
        if len(b) < 2 or len(t) < 2:
            continue
        out[qid] = (float(np.mean(b) - np.mean(t)), float(np.mean(b)))
    return out


def boot(d: np.ndarray, n: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    v = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    subjects = json.load(open("data/pairs/subjects.json"))
    all_diffs, rows = [], []
    print("\n" + "=" * 78)
    print("WITHIN EACH PAIR: how much more is the ACTION question damaged?")
    print("   (positive means the action half was damaged more)")
    print("=" * 78)
    for prefix, nice in FAMILIES:
        fam_diffs = []
        for a, ad in enumerate(ADAPTERS):
            d = damage(f"{prefix}_{ad}")
            diffs, bf, ba = [], [], []
            for i in range(len(subjects)):
                f, act = f"p{i:02d}_fact", f"p{i:02d}_action"
                if f in d and act in d:
                    diffs.append(d[act][0] - d[f][0])
                    bf.append(d[f][1]); ba.append(d[act][1])
                    rows.append((i, prefix, d[act][0] - d[f][0]))
            diffs = np.array(diffs)
            lo, hi = boot(diffs)
            w = wilcoxon(diffs).pvalue if len(diffs) > 5 and np.any(diffs != 0) else np.nan
            share = float((diffs > 0).mean())
            print(f"   {nice:14s} {SHORT[a]:10s} {len(diffs):2d} usable pairs   "
                  f"action minus fact {diffs.mean():+6.2f} [{lo:+.2f},{hi:+.2f}]   "
                  f"action worse in {share:.0%} of pairs   p={w:.4f}")
            print(f"                            untouched model scored the fact half "
                  f"{np.mean(bf):.1f} and the action half {np.mean(ba):.1f}")
            fam_diffs.append(diffs)
            all_diffs.append(diffs)
        pooled = np.concatenate(fam_diffs)
        lo, hi = boot(pooled)
        print(f"   {nice:14s} {'ALL THREE':10s} {len(pooled):2d} pair-scores  "
              f"action minus fact {pooled.mean():+6.2f} [{lo:+.2f},{hi:+.2f}]\n")

    pooled = np.concatenate(all_diffs)
    lo, hi = boot(pooled)
    verdict = ("the action half is damaged more" if lo > 0 else
               "the fact half is damaged more" if hi < 0 else
               "no difference between the halves")
    print("=" * 78)
    print(f"   EVERYTHING POOLED: {len(pooled)} pair-scores   "
          f"action minus fact {pooled.mean():+.2f} [{lo:+.2f},{hi:+.2f}]")
    print(f"   -> {verdict}")
    print("=" * 78)

    by_subject: dict[int, list[float]] = {}
    for i, _, v in rows:
        by_subject.setdefault(i, []).append(v)
    order = sorted(by_subject, key=lambda i: -np.mean(by_subject[i]))
    print("\n   subjects where asking for action hurt most:")
    for i in order[:6]:
        print(f"     {np.mean(by_subject[i]):+6.1f}  {subjects[i]['subject']}")
    print("   subjects where it hurt least:")
    for i in order[-4:]:
        print(f"     {np.mean(by_subject[i]):+6.1f}  {subjects[i]['subject']}")


if __name__ == "__main__":
    main()
