"""Pooled version of the fair-cost comparison: both sides get a whole answer."""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr

from analyze import ground_truth
from pooled import TAGS, clustered, pooled_rho


def build() -> list[dict]:
    runs = []
    for tag in TAGS:
        sig = json.load(open(f"results/{tag}_signals.json"))
        forced = json.load(open(f"results/{tag}_forced.json"))
        rank = np.load(f"results/{tag}_ranksignals.npz", allow_pickle=True)
        drops, _, _ = ground_truth(tag)
        ids_all = [r["id"] for r in sig if r["id"] in drops]
        pos = {q: j for j, q in enumerate(ids_all)}
        keep = [i for i, r in enumerate(sig) if r["id"] in drops and r["id"] in forced]
        sel = np.array([pos[sig[i]["id"]] for i in keep])
        nf = np.array([forced[sig[i]["id"]]["norm_forced"] for i in keep])
        y = np.array([drops[sig[i]["id"]] for i in keep])
        best = int(np.nanargmax([spearmanr(nf[:, L], y).statistic for L in range(nf.shape[1])]))
        runs.append({
            "y": y,
            "kl": np.array([forced[sig[i]["id"]]["kl_forced"] for i in keep]),
            "kl_cheap": np.array([sig[i]["kl"] for i in keep]),
            "in_forced": nf[:, best],
            "in_fit_cheap": rank["in_fit"][sel],
        })
    return runs


def main() -> None:
    runs = build()
    idx = [np.arange(len(r["y"])) for r in runs]
    n = sum(len(i) for i in idx)
    print(f"\n=== both sides allowed to look at a whole answer ({n} question-slots) ===")
    for name, key in (("outputs, over the answer", "kl"), ("internals, over the answer", "in_forced")):
        rho = pooled_rho(runs, key, idx)
        per = "  ".join(f"{spearmanr(r[key][i], r['y'][i]).statistic:+.2f}" for r, i in zip(runs, idx))
        line = f"  {name:30s} {rho:+.3f}   per fine-tune: {per}"
        if key != "kl":
            g, lo, hi = clustered(runs, key, "kl", False)
            line += f"   vs outputs {g:+.3f} [{lo:+.3f},{hi:+.3f}]  " + \
                    ("BEATS outputs" if lo > 0 else "not separated")
        print(line)

    print("\n=== is one cheap look inside worth a whole generation outside? ===")
    rho_cheap = pooled_rho(runs, "in_fit_cheap", idx)
    rho_exp = pooled_rho(runs, "kl", idx)
    g, lo, hi = clustered(runs, "in_fit_cheap", "kl", False)
    verdict = ("the cheap internal signal is ahead" if lo > 0 else
               "the expensive output signal is ahead" if hi < 0 else
               "they are indistinguishable")
    print(f"  internals, one forward pass on the question   {rho_cheap:+.3f}")
    print(f"  outputs, a whole generated answer             {rho_exp:+.3f}")
    print(f"  difference {g:+.3f} [{lo:+.3f},{hi:+.3f}]  ->  {verdict}")


if __name__ == "__main__":
    main()
