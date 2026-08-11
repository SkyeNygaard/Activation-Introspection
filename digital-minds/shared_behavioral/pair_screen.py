"""Admission rule for binding task-family preferences.

Competence is NOT decided here. It is established per family, before pairs exist,
by family_screen.py on a disjoint item set. These four counterbalanced variants
measure one thing: preference stability. Correctness is carried as a diagnostic
covariate so a treatment-induced competence change is still visible.

Branch-specific admission, because the two branches want different things:

- Branch 18 (path dependence) needs a baseline preference that can subsequently
  MOVE, so it wants 4/4 -- or independent strong evidence of a baseline.
- Branch 19 (preference self-knowledge) has robustness itself as the dependent
  variable. Admitting only 4/4 truncates that variable before asking whether the
  model knows how robust its preferences are, so it takes 3/4 and deliberately
  retains a spectrum of stability.
"""
from __future__ import annotations
import pandas as pd

N_VARIANTS = 4
STABILITY_RULES = {
    # branch -> minimum number of the 4 variants agreeing on one canonical choice
    "18": 4,
    "19": 3,
    "default": 3,
}


def stability_threshold(branch: str = "default") -> int:
    if branch not in STABILITY_RULES:
        raise ValueError(f"no admission rule for branch {branch!r}")
    return STABILITY_RULES[branch]


def screen_pairs(df: pd.DataFrame, *, branch: str = "default",
                 eligible_families=None):
    """Admit pairs on preference stability alone.

    `eligible_families`: families that already passed the independent competence
    screen. Pairs touching anything else are rejected before stability is read,
    which is where competence belongs in the pipeline.
    """
    req = {"pair_id", "canonical_choice", "task_correct", "valid_choice"}
    if not req <= set(df):
        raise ValueError(req - set(df))
    need = stability_threshold(branch)
    has_families = {"family_A", "family_B"} <= set(df)
    if eligible_families is not None and not has_families:
        raise ValueError("eligible_families given but df lacks family_A/family_B")
    elig = set(eligible_families) if eligible_families is not None else None

    rows = []
    for pair, g in df.groupby("pair_id"):
        valid = g[g.valid_choice]
        rec = {"pair_id": pair, "branch": branch, "required_agreement": need}

        if elig is not None:
            fams = {str(g.family_A.iloc[0]), str(g.family_B.iloc[0])}
            if not fams <= elig:
                rows.append({**rec, "admitted": False,
                             "reason": f"family failed competence screen: {sorted(fams-elig)}"})
                continue

        if len(valid) != N_VARIANTS:
            rows.append({**rec, "admitted": False,
                         "reason": f"needs {N_VARIANTS} valid admission variants"})
            continue

        counts = valid.canonical_choice.value_counts()
        agree = int(counts.iloc[0])
        rows.append({
            **rec,
            "preferred": str(counts.index[0]),
            "agreement": agree,
            "stability": agree / N_VARIANTS,
            # Reported, never the selector. A single formatting miss no longer
            # excludes a pair; a systematic drop still shows up here.
            "task_correct": float(valid.task_correct.mean()),
            "admitted": agree >= need,
            "reason": "pass" if agree >= need else "unstable_preference",
        })
    return pd.DataFrame(rows)


def stability_spectrum(screened: pd.DataFrame) -> dict:
    """Branch 19 wants variation here, not a ceiling. Check it survived."""
    adm = screened[screened.admitted]
    if adm.empty:
        return {"n_admitted": 0}
    counts = adm.agreement.value_counts().to_dict()
    return {
        "n_admitted": int(len(adm)),
        "agreement_counts": {int(k): int(v) for k, v in sorted(counts.items())},
        "fraction_at_ceiling": float((adm.agreement == N_VARIANTS).mean()),
        "retains_spectrum": bool(adm.agreement.nunique() > 1),
    }
