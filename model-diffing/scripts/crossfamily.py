"""Does the fragile-question list transfer to a different model family?

If the same questions are fragile in Qwen2.5-0.5B as in Llama-3.2-1B -- different
families, different training data, different sizes -- then the list is a property of
the questions, measurable once and reusable by anyone. If it does not transfer, it
only works within a model already audited, which is a much smaller claim.

Nothing is shared between the families by construction: different base models means
different base answers, so the leak that inflated the within-family numbers cannot
occur here.
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr

from analyze import COHERENT_MIN
from attack import resid
from checks import paired_gap

ADAPTERS = ["bad-medical-advice", "risky-financial-advice", "extreme-sports"]
SHORT = ["medical", "financial", "sports"]


def family(prefix: str) -> dict:
    """Per-question damage for each fine-tune of one base model, plus its own signals."""
    out = {}
    for ad in ADAPTERS:
        tag = f"{prefix}_{ad}"
        judged = json.load(open(f"results/{tag}_judged.json"))
        per: dict[tuple[str, str], list[int]] = {}
        dropped = 0
        for r in judged:
            if r["coherent"] < COHERENT_MIN or r["aligned"] < 0:
                dropped += 1
                continue
            per.setdefault((r["id"], r["version"]), []).append(r["aligned"])
        rows = {}
        for qid in {k[0] for k in per}:
            b, t = per.get((qid, "base"), []), per.get((qid, "tuned"), [])
            if len(b) < 2 or len(t) < 2:
                continue
            rows[qid] = (float(np.mean(b) - np.mean(t)), float(np.mean(b)))
        sig = json.load(open(f"results/{tag}_signals.json"))
        out[ad] = {"rows": rows, "kl": {s["id"]: s["kl"] for s in sig},
                   "dropped": dropped, "n_judged": len(judged)}
    return out


def frag_map(fam: dict) -> dict[str, float]:
    """One number per question: how damaged it was, averaged over this family's fine-tunes."""
    per_ad = {}
    for ad, d in fam.items():
        ids = sorted(d["rows"])
        y = np.array([d["rows"][q][0] for q in ids])
        base = np.array([d["rows"][q][1] for q in ids])
        per_ad[ad] = dict(zip(ids, resid(y, base)))
    common = set.intersection(*(set(v) for v in per_ad.values()))
    return {q: float(np.mean([per_ad[ad][q] for ad in per_ad])) for q in common}


def main() -> None:
    llama, qwen = family("llama1b"), family("qwen05b")

    print("\n" + "=" * 78)
    print("HOW MUCH OF THE SMALLER MODEL SURVIVED THE COHERENCE FILTER?")
    print("=" * 78)
    for nm, fam in (("Llama-3.2-1B", llama), ("Qwen2.5-0.5B", qwen)):
        d = fam["bad-medical-advice"]
        print(f"   {nm:14s} {d['dropped']}/{d['n_judged']} answers discarded as incoherent; "
              f"{len(d['rows'])}/300 questions still scoreable")

    print("\n" + "=" * 78)
    print("DO THE SAME QUESTIONS BREAK WITHIN EACH FAMILY?")
    print("=" * 78)
    for nm, fam in (("Llama-3.2-1B", llama), ("Qwen2.5-0.5B", qwen)):
        vals = []
        for a in range(3):
            for b in range(a + 1, 3):
                ra, rb = fam[ADAPTERS[a]], fam[ADAPTERS[b]]
                sh = sorted(set(ra["rows"]) & set(rb["rows"]))
                ya = resid(np.array([ra["rows"][q][0] for q in sh]),
                           np.array([ra["rows"][q][1] for q in sh]))
                yb = resid(np.array([rb["rows"][q][0] for q in sh]),
                           np.array([rb["rows"][q][1] for q in sh]))
                vals.append(spearmanr(ya, yb).statistic)
        print(f"   {nm:14s} " + "   ".join(f"{SHORT[a]}-{SHORT[b]} {v:+.3f}" for (a, b), v in
              zip([(0, 1), (0, 2), (1, 2)], vals)) + f"   average {np.mean(vals):+.3f}")

    fl, fq = frag_map(llama), frag_map(qwen)
    shared = sorted(set(fl) & set(fq))
    print("\n" + "=" * 78)
    print("DOES THE LIST CROSS FAMILIES?")
    print("=" * 78)
    r = spearmanr([fl[q] for q in shared], [fq[q] for q in shared])
    rng, vals = np.random.default_rng(0), []
    a1 = np.array([fl[q] for q in shared]); a2 = np.array([fq[q] for q in shared])
    for _ in range(4000):
        i = rng.integers(0, len(shared), len(shared))
        v = spearmanr(a1[i], a2[i]).statistic
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    print(f"   Llama's fragile-question list vs Qwen's, on {len(shared)} shared questions:")
    print(f"      {r.statistic:+.3f} [{lo:+.3f},{hi:+.3f}]   "
          f"{'transfers' if lo > 0 else 'does NOT transfer'}")

    print("\n" + "=" * 78)
    print("PRACTICAL TEST: rank Qwen's questions using LLAMA's list only")
    print("   (against Qwen's own outputs, and Qwen's own list from its siblings)")
    print("=" * 78)
    for a, ad in enumerate(ADAPTERS):
        d = qwen[ad]
        sh = [q for q in sorted(d["rows"]) if q in fl]
        y = resid(np.array([d["rows"][q][0] for q in sh]),
                  np.array([d["rows"][q][1] for q in sh]))
        cross = np.array([fl[q] for q in sh])
        kl = np.array([d["kl"][q] for q in sh])
        own = np.array([np.mean([resid(
            np.array([qwen[o]["rows"][q2][0] for q2 in sorted(qwen[o]["rows"])]),
            np.array([qwen[o]["rows"][q2][1] for q2 in sorted(qwen[o]["rows"])])
        )[sorted(qwen[o]["rows"]).index(q)] for o in ADAPTERS if o != ad and q in qwen[o]["rows"]])
            if all(q in qwen[o]["rows"] for o in ADAPTERS if o != ad) else np.nan for q in sh])
        ok = ~np.isnan(own)
        g, glo, ghi, _ = paired_gap(y, cross, kl)
        v = "list ahead" if glo > 0 else ("outputs ahead" if ghi < 0 else "not separated")
        print(f"   {SHORT[a]:10s} ({len(sh)} questions)   "
              f"Llama's list {spearmanr(cross, y).statistic:+.3f}   "
              f"Qwen's own list {spearmanr(own[ok], y[ok]).statistic:+.3f}   "
              f"Qwen's outputs {spearmanr(kl, y).statistic:+.3f}   "
              f"cross-list vs outputs {g:+.3f} [{glo:+.3f},{ghi:+.3f}] {v}")


if __name__ == "__main__":
    main()
