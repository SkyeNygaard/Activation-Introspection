"""Analysis scaffold for welfare × Assistant-axis 2x2 factorial."""
from __future__ import annotations
import pandas as pd

def factorial_contrasts(df: pd.DataFrame,outcome: str):
    means=df.groupby(["W","A"])[outcome].mean()
    mpp=float(means.loc[(1,1)]); mpm=float(means.loc[(1,-1)])
    mmp=float(means.loc[(-1,1)]); mmm=float(means.loc[(-1,-1)])
    return {
        "welfare_main":((mpp+mpm)-(mmp+mmm))/2,
        "assistant_main":((mpp+mmp)-(mpm+mmm))/2,
        "interaction":(mpp-mpm-mmp+mmm)/2,
    }
