"""Analyze paired instance/cache persistence after transient intervention."""
from __future__ import annotations
import pandas as pd
import numpy as np

def trajectory_effects(df: pd.DataFrame, outcome="welfare_projection"):
    req={"lineage_id","arm","turn","mode",outcome}
    if not req <= set(df): raise ValueError(req-set(df))
    rows=[]
    for (lineage,mode,turn),g in df.groupby(["lineage_id","mode","turn"]):
        if set(g.arm)!={-1,1} or len(g)!=2:
            raise ValueError((lineage,mode,turn))
        pos=float(g.loc[g.arm==1,outcome].iloc[0])
        neg=float(g.loc[g.arm==-1,outcome].iloc[0])
        rows.append({"lineage_id":lineage,"mode":mode,"turn":turn,"delta":pos-neg})
    out=pd.DataFrame(rows)
    return out.groupby(["mode","turn"]).delta.agg(["mean","std","count"]).reset_index()
