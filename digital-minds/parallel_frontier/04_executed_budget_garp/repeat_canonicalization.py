"""Canonicalize relabel/order repeats for skip-budget observations."""
from __future__ import annotations
import pandas as pd

def display_budget(canonical_pA,canonical_pB,budget,*,swap_families:bool):
    """Return displayed prices for labels A/B while preserving canonical menu."""
    if swap_families:
        return {"pA":canonical_pB,"pB":canonical_pA,"budget":budget}
    return {"pA":canonical_pA,"pB":canonical_pB,"budget":budget}

def canonicalize_choice(skip_A,skip_B,*,swap_families:bool):
    """Displayed A/B -> canonical family-A/family-B skip coordinates."""
    if swap_families:
        return (int(skip_B),int(skip_A))
    return (int(skip_A),int(skip_B))

def aggregate_repeats(df: pd.DataFrame, min_fraction=.75):
    req={
        "budget_id","canonical_pA","canonical_pB","budget",
        "skip_A","skip_B","swap_families","valid_choice"
    }
    if not req<=set(df): raise ValueError(req-set(df))
    rows=[]
    for bid,g in df.groupby("budget_id"):
        valid=g[g.valid_choice]
        canon=[
            canonicalize_choice(r.skip_A,r.skip_B,swap_families=bool(r.swap_families))
            for r in valid.itertuples()
        ]
        if not canon:
            rows.append({"budget_id":bid,"stable":False,"reason":"no_valid_choices"})
            continue
        counts=pd.Series(canon).value_counts()
        top=tuple(counts.index[0])
        fraction=float(counts.iloc[0]/len(canon))
        r0=valid.iloc[0]
        rows.append({
            "budget_id":bid,
            "prices":(int(r0.canonical_pA),int(r0.canonical_pB)),
            "budget":int(r0.budget),
            "choice":top,
            "valid_repeats":len(canon),
            "stability_fraction":fraction,
            "stable":fraction>=min_fraction,
            "reason":"pass" if fraction>=min_fraction else "unstable",
        })
    return pd.DataFrame(rows)
