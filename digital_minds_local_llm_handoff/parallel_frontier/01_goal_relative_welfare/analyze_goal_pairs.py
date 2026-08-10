"""Paired analysis for goal-relative projection experiment."""
from __future__ import annotations
import pandas as pd
import numpy as np

def paired_effect(df: pd.DataFrame, outcome="welfare_projection"):
    req={"pair_id","success",outcome}
    if not req <= set(df):
        raise ValueError(f"missing {req-set(df)}")
    rows=[]
    for pair,g in df.groupby("pair_id"):
        if set(g["success"])!={False,True} or len(g)!=2:
            raise ValueError(f"{pair}: need exact success/failure pair")
        s=float(g.loc[g.success,outcome].iloc[0])
        f=float(g.loc[~g.success,outcome].iloc[0])
        rows.append(s-f)
    x=np.asarray(rows)
    return {
        "mean_success_minus_failure":float(x.mean()),
        "sd_pairs":float(x.std(ddof=1)),
        "n_pairs":len(x),
    }
