"""Does the outside signal catch up when it is allowed to look at a whole answer?"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import spearmanr

from analyze import ground_truth
from checks import paired_gap

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
a = ap.parse_args()

sig = json.load(open(f"results/{a.tag}_signals.json"))
forced = json.load(open(f"results/{a.tag}_forced.json"))
rank = np.load(f"results/{a.tag}_ranksignals.npz", allow_pickle=True)
drops, _, _ = ground_truth(a.tag)

keep = [i for i, r in enumerate(sig) if r["id"] in drops and r["id"] in forced]
y = np.array([drops[sig[i]["id"]] for i in keep])
klf = np.array([forced[sig[i]["id"]]["kl_forced"] for i in keep])
nf = np.array([forced[sig[i]["id"]]["norm_forced"] for i in keep])
kl0 = np.array([sig[i]["kl"] for i in keep])
# The prompt-only signals were computed on a slightly larger set; line them up.
ids_all = [sig[i]["id"] for i, r in enumerate(sig) if r["id"] in drops]
pos = {q: j for j, q in enumerate(ids_all)}
sel = np.array([pos[sig[i]["id"]] for i in keep])
in_fit = rank["in_fit"][sel]

best_f = int(np.nanargmax([spearmanr(nf[:, L], y).statistic for L in range(nf.shape[1])]))
print(f"\n{a.tag}: {len(y)} questions\n")
print("  cost tier 1 -- one forward pass on the question, nothing generated")
print(f"    outputs, one number                       {spearmanr(kl0, y).statistic:+.3f}")
print(f"    internals, fitted reader                  {spearmanr(in_fit, y).statistic:+.3f}")
print("\n  cost tier 2 -- replay a whole generated answer through both models")
print(f"    outputs, averaged over the answer         {spearmanr(klf, y).statistic:+.3f}")
print(f"    internals, averaged over the answer       "
      f"{spearmanr(nf[:, best_f], y).statistic:+.3f}  [depth {best_f}]")

nfb = nf[:, best_f]
g, lo, hi, _ = paired_gap(y, in_fit, klf)
verdict = "yes" if lo > 0 else ("no" if hi < 0 else "not separated")
print(f"\n  does the cheap internal signal still beat the expensive output one? "
      f"{g:+.3f} [{lo:+.3f},{hi:+.3f}]  -> {verdict}")
g2, lo2, hi2, _ = paired_gap(y, klf, kl0)
print(f"  did paying for a generation help the output side at all?           "
      f"{g2:+.3f} [{lo2:+.3f},{hi2:+.3f}]")
g3, lo3, hi3, _ = paired_gap(y, nfb, klf)
v3 = "yes" if lo3 > 0 else ("no" if hi3 < 0 else "not separated")
print(f"  at that higher cost, do internals still beat outputs?              "
      f"{g3:+.3f} [{lo3:+.3f},{hi3:+.3f}]  -> {v3}")
