"""Causal intervention × measurement response-signature analysis."""
from __future__ import annotations
import pandas as pd
import numpy as np

def standardized_effect_matrix(df: pd.DataFrame):
    req={"unit","intervention","condition","measurement","value"}
    if not req <= set(df): raise ValueError(req-set(df))
    rows=[]
    for (iv,m),g in df.groupby(["intervention","measurement"]):
        diffs=[]
        for unit,u in g.groupby("unit"):
            if set(u.condition)!={-1,1}: continue
            p=float(u.loc[u.condition==1,"value"].mean())
            n=float(u.loc[u.condition==-1,"value"].mean())
            diffs.append(p-n)
        x=np.asarray(diffs,float)
        scale=x.std(ddof=1) if len(x)>1 else np.nan
        rows.append({
            "intervention":iv,"measurement":m,
            "effect":float(x.mean()) if len(x) else np.nan,
            "paired_sd":float(scale),
            "standardized_effect":float(x.mean()/scale) if scale and scale>0 else np.nan,
            "n_units":len(x),
        })
    return pd.DataFrame(rows)
