"""Four free checks on the headline, two of which could overturn it.

1. fragility     -- do the same questions break under every fine-tune?
2. floor effect  -- is the damage measure partly "it had further to fall"?
3. transfer      -- does a reader calibrated on one fine-tune flag another?
4. direction     -- did the three readers find the same axis?
plus a harder target: the outright-bad answers rather than mild degradation.
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze import COHERENT_MIN, MISALIGNED_MAX, WORST_FRAC, recall_at
from checks import paired_gap
from pooled import TAGS

SHORT = ["medical", "financial", "sports"]


def per_question(tag: str) -> dict[str, dict]:
    """Damage, plus the level it started from and the strict outright-bad label."""
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
        out[qid] = {"drop": float(np.mean(b) - np.mean(t)), "base": float(np.mean(b)),
                    "tuned": float(np.mean(t)),
                    "bad": int(min(t) < MISALIGNED_MAX <= min(b))}
    return out


def load():
    runs = []
    for tag in TAGS:
        sig = json.load(open(f"results/{tag}_signals.json"))
        D = np.load(f"results/{tag}_deltas.npz")["deltas"]
        rank = np.load(f"results/{tag}_ranksignals.npz", allow_pickle=True)
        pq = per_question(tag)
        keep = [i for i, r in enumerate(sig) if r["id"] in pq]
        runs.append({
            "tag": tag, "ids": [sig[i]["id"] for i in keep],
            "y": np.array([pq[sig[i]["id"]]["drop"] for i in keep]),
            "base": np.array([pq[sig[i]["id"]]["base"] for i in keep]),
            "bad": np.array([pq[sig[i]["id"]]["bad"] for i in keep]),
            "kl": np.array([sig[i]["kl"] for i in keep]),
            "in_fit": rank["in_fit"], "D": D[keep],
        })
    return runs


def resid(x: np.ndarray, on: np.ndarray) -> np.ndarray:
    return x - np.poly1d(np.polyfit(on, x, 1))(on)


def main() -> None:
    runs = load()
    idmap = [{q: j for j, q in enumerate(r["ids"])} for r in runs]

    print("\n" + "=" * 78)
    print("1. ARE SOME QUESTIONS JUST FRAGILE?  (if yes, no second model is needed)")
    print("=" * 78)
    for a in range(3):
        for b in range(3):
            if a >= b:
                continue
            shared = [q for q in runs[a]["ids"] if q in idmap[b]]
            ya = runs[a]["y"][[idmap[a][q] for q in shared]]
            yb = runs[b]["y"][[idmap[b][q] for q in shared]]
            print(f"   damage under {SHORT[a]:10s} vs under {SHORT[b]:10s}  "
                  f"{spearmanr(ya, yb).statistic:+.3f}   ({len(shared)} shared questions)")

    print("\n   used as a ranking signal -- rank one fine-tune's questions by how damaged")
    print("   they were under the OTHER TWO, and compare with the internal reader:")
    for a in range(3):
        others = [b for b in range(3) if b != a]
        shared = [q for q in runs[a]["ids"] if all(q in idmap[b] for b in others)]
        ia = [idmap[a][q] for q in shared]
        y = runs[a]["y"][ia]
        frag = np.mean([runs[b]["y"][[idmap[b][q] for q in shared]] for b in others], axis=0)
        inr = runs[a]["in_fit"][ia]
        g, lo, hi, _ = paired_gap(y, inr, frag)
        verdict = "internals ahead" if lo > 0 else ("fragility ahead" if hi < 0 else "not separated")
        print(f"   {SHORT[a]:10s}  fragility {spearmanr(frag, y).statistic:+.3f}   "
              f"internals {spearmanr(inr, y).statistic:+.3f}   "
              f"gap {g:+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")

    print("\n" + "=" * 78)
    print("2. IS THE DAMAGE MEASURE PARTLY 'IT HAD FURTHER TO FALL'?")
    print("=" * 78)
    for a, r in enumerate(runs):
        y, base, kl, inr = r["y"], r["base"], r["kl"], r["in_fit"]
        print(f"   {SHORT[a]:10s} untouched model's own score vs the drop  "
              f"{spearmanr(base, y).statistic:+.3f}")
        yr = resid(y, base)
        g, lo, hi, _ = paired_gap(yr, inr, kl)
        verdict = "internals ahead" if lo > 0 else ("outputs ahead" if hi < 0 else "not separated")
        print(f"              with that removed:  outputs {spearmanr(kl, yr).statistic:+.3f}   "
              f"internals {spearmanr(inr, yr).statistic:+.3f}   "
              f"gap {g:+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")

    print("\n" + "=" * 78)
    print("3. DOES A READER CALIBRATED ON ONE FINE-TUNE FLAG A DIFFERENT ONE?")
    print("=" * 78)
    depth = 10  # mid-depth, where the advantage lives; fixed, not tuned per pair
    fitted = []
    for r in runs:
        cut = np.quantile(r["y"], 1 - WORST_FRAC)
        m = make_pipeline(StandardScaler(), PCA(n_components=12, random_state=0),
                          LogisticRegression(max_iter=2000))
        m.fit(r["D"][:, depth], (r["y"] > cut).astype(int))
        fitted.append(m)
    print(f"   (reader fitted at depth {depth}, applied to another fine-tune's internals)")
    for a in range(3):
        row = []
        for b in range(3):
            s = fitted[a].predict_proba(runs[b]["D"][:, depth])[:, 1]
            row.append(spearmanr(s, runs[b]["y"]).statistic)
        marks = "   ".join(f"{SHORT[b][:4]} {row[b]:+.3f}" + ("*" if a == b else " ")
                           for b in range(3))
        off = np.mean([row[b] for b in range(3) if b != a])
        print(f"   fitted on {SHORT[a]:10s} -> {marks}    average on the other two {off:+.3f}")
    print("   (* = tested on the fine-tune it was fitted on, so not a fair number)")

    print("\n" + "=" * 78)
    print("4. DID THE THREE READERS FIND THE SAME AXIS?")
    print("=" * 78)
    dirs = []
    for m in fitted:
        pca, lr = m.named_steps["pca"], m.named_steps["logisticregression"]
        v = lr.coef_[0] @ pca.components_
        dirs.append(v / np.linalg.norm(v))
    for a in range(3):
        for b in range(a + 1, 3):
            print(f"   {SHORT[a]:10s} vs {SHORT[b]:10s}  overlap {float(dirs[a] @ dirs[b]):+.3f}")
    rng = np.random.default_rng(0)
    rnd = [float(abs(x)) for x in
           [(lambda u, v: u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))(
               rng.normal(size=dirs[0].size), rng.normal(size=dirs[0].size)) for _ in range(200)]]
    print(f"   two unrelated directions in this space overlap by about "
          f"{np.mean(rnd):.3f} (up to {np.max(rnd):.3f})")

    print("\n" + "=" * 78)
    print("5. HARDER TARGET: the outright-bad answers, not mild degradation")
    print("=" * 78)
    for a, r in enumerate(runs):
        bad = r["bad"].astype(float)
        if bad.sum() < 8:
            print(f"   {SHORT[a]:10s} only {int(bad.sum())} such questions; skipped")
            continue
        g, lo, hi, _ = paired_gap(bad, r["in_fit"], r["kl"])
        verdict = "internals ahead" if lo > 0 else ("outputs ahead" if hi < 0 else "not separated")
        print(f"   {SHORT[a]:10s} {int(bad.sum())} outright-bad questions   "
              f"outputs {spearmanr(r['kl'], bad).statistic:+.3f}   "
              f"internals {spearmanr(r['in_fit'], bad).statistic:+.3f}   "
              f"gap {g:+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")
        print(f"              catch@20%: random 0.20   outputs "
              f"{recall_at(bad, r['kl'], 0.2):.2f}   internals "
              f"{recall_at(bad, r['in_fit'], 0.2):.2f}")


if __name__ == "__main__":
    main()
