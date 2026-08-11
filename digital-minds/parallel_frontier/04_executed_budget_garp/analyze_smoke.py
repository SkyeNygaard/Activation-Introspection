"""DEV promotion checks for skip-budget experiment."""
from __future__ import annotations
import pandas as pd
import numpy as np

def smoke_summary(df: pd.DataFrame):
    req={"budget_id","pA","pB","skip_A","skip_B","valid","pareto_maximal","task_correct"}
    if not req<=set(df): raise ValueError(req-set(df))

    valid=float(df.valid.mean())
    maximal=float(df.pareto_maximal.mean())
    competence=float(df.task_correct.dropna().mean()) if df.task_correct.notna().any() else float("nan")
    unique=len(set(zip(df.skip_A,df.skip_B)))

    good=df[df.valid & df.pareto_maximal].copy()
    if len(good)>=2:
        good["log_price_ratio"]=np.log(good.pA/good.pB)
        good["A_share"]=good.skip_A/(good.skip_A+good.skip_B).replace(0,np.nan)
        corr=float(good.log_price_ratio.corr(good.A_share))
    else:
        corr=float("nan")

    # As A becomes more expensive relative to B, a normal interior demand response
    # is a lower A share; hence negative correlation is the expected diagnostic.
    gates={
        "valid_rate_gte_0.95":valid>=.95,
        "pareto_maximal_rate_gte_0.90":maximal>=.90,
        "task_correct_gte_0.90":competence>=.90,
        "at_least_two_unique_bundles":unique>=2,
        "relative_price_response_negative":bool(np.isfinite(corr) and corr<0),
    }
    return {
        "valid_rate":valid,"pareto_maximal_rate":maximal,
        "task_correct_rate":competence,"unique_bundles":unique,
        "price_Ashare_correlation":corr,
        "gates":{**gates,"passed":all(gates.values())},
    }
