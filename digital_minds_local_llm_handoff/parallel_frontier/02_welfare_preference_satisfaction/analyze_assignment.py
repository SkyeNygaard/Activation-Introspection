"""Paired preference-satisfaction -> welfare analysis."""
from __future__ import annotations
import pandas as pd
import numpy as np

def summarize(df: pd.DataFrame, outcome="welfare_projection"):
    req={"pair_id","condition",outcome,"task_correct"}
    if not req <= set(df): raise ValueError(req-set(df))
    diffs=[]
    for pair,g in df.groupby("pair_id"):
        if set(g.condition)!={"preferred","dispreferred"}:
            raise ValueError(f"{pair}: assignment pair incomplete")
        a=float(g.loc[g.condition=="preferred",outcome].iloc[0])
        b=float(g.loc[g.condition=="dispreferred",outcome].iloc[0])
        diffs.append(a-b)
    acc={c:float(g.task_correct.mean()) for c,g in df.groupby("condition")}
    x=np.array(diffs)
    return {
        "preferred_minus_dispreferred":float(x.mean()),
        "sd_pairs":float(x.std(ddof=1)),
        "n_pairs":len(x),
        "accuracy_by_condition":acc,
        "accuracy_gap":acc["preferred"]-acc["dispreferred"],
    }
