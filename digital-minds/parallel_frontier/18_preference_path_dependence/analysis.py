"""Carrier/pair-level analysis for preference path dependence."""
from __future__ import annotations
import pandas as pd
import numpy as np

def pair_effects(df: pd.DataFrame):
    req={"pair_id","assignment","dose","context_mode","choose_A"}
    if not req<=set(df): raise ValueError(req-set(df))
    rows=[]
    for (pair,dose,mode),g in df.groupby(["pair_id","dose","context_mode"]):
        a=g[g.assignment=="A"].choose_A.astype(float)
        b=g[g.assignment=="B"].choose_A.astype(float)
        if len(a)==0 or len(b)==0: raise ValueError("missing randomized arm")
        rows.append({
            "pair_id":pair,
            "dose":int(dose),
            "context_mode":mode,
            # Negative means executing A makes A less likely than executing B:
            # relative satiation / repetition aversion.
            "A_after_A_minus_A_after_B":float(a.mean()-b.mean()),
        })
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame):
    pe=pair_effects(df)
    out={}
    for (dose,mode),g in pe.groupby(["dose","context_mode"]):
        x=g["A_after_A_minus_A_after_B"].to_numpy(float)
        out[f"dose{dose}:{mode}"]={
            "mean":float(x.mean()),
            "sd_pairs":float(x.std(ddof=1)) if len(x)>1 else None,
            "n_pairs":len(x),
        }

    # Difference between rich full history and blank reset at dose 3.
    def m(mode,dose=3):
        x=pe[(pe.context_mode==mode)&(pe.dose==dose)]
        return float(x.A_after_A_minus_A_after_B.mean())
    out["history_specificity_dose3"]={
        "full_minus_blank":m("full_history")-m("blank_reset"),
        "summary_minus_blank":m("summary_only")-m("blank_reset"),
        "full_minus_summary":m("full_history")-m("summary_only"),
    }

    # Dose response in full history.
    out["dose_response_full_history"]={
        "dose3_minus_dose1":m("full_history",3)-m("full_history",1)
    }
    return out
