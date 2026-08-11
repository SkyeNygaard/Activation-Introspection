"""Calibration analysis for predicted vs realized preference robustness.

Headline is a proper score, not MAE. MAE rewards a pure bias shift: a forecaster
that is perfectly ordered but uniformly 15 points too high looks bad on MAE and
is fixed by subtracting a constant, so "structured elicitation cut MAE" can mean
nothing more than "structured elicitation happened to be centred". The three
questions are separate and are reported separately:

  brier                   -- proper score: is the forecast good?
  calibration_bias/slope  -- is it systematically over/under-confident?
  correlation / rank      -- does it know WHICH preferences are fragile?

A model can correlate perfectly with realized robustness while being wildly
overconfident everywhere, so discrimination alone is not the headline either.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

N_VARIANTS = 12


def item_table(forecasts: pd.DataFrame, choices: pd.DataFrame):
    f_req={"pair_id","forecast_method","predicted_robustness"}
    c_req={"pair_id","variant_id","chose_canonical"}
    if not f_req<=set(forecasts): raise ValueError(f_req-set(forecasts))
    if not c_req<=set(choices): raise ValueError(c_req-set(choices))

    actual=(
        choices.groupby("pair_id").chose_canonical.mean()
        .rename("actual_robustness").reset_index()
    )
    counts=choices.groupby("pair_id").size()
    if not (counts==N_VARIANTS).all():
        raise ValueError(f"each pair needs exactly {N_VARIANTS} fixed variants")

    out=forecasts.merge(actual,on="pair_id",validate="many_to_one")
    if ((out.predicted_robustness<0)|(out.predicted_robustness>1)).any():
        raise ValueError("forecast outside [0,1]")
    out["error"]=out.predicted_robustness-out.actual_robustness
    out["abs_error"]=out.error.abs()
    out["sq_error"]=out.error**2
    return out


def _brier(pred: np.ndarray, actual: np.ndarray) -> float:
    """Item-level Brier over the 12 binary variant outcomes of each pair.

    With a forecast constant within a pair, mean_i (p - y_i)^2 collapses to
    (p - a)^2 + a(1 - a) exactly, since y is 0/1. The second term is the pair's
    own variability and no forecaster can beat it -- reported as the floor.
    """
    return float(np.mean((pred-actual)**2 + actual*(1-actual)))


def _slope(pred: pd.Series, actual: pd.Series):
    """OLS slope of realized on predicted. 1.0 = correctly scaled spread."""
    if len(pred)<3 or pred.nunique()<2: return None
    return float(np.polyfit(pred,actual,1)[0])


def _stats(pred: np.ndarray, actual: np.ndarray) -> dict:
    brier=_brier(pred,actual)
    floor=float(np.mean(actual*(1-actual)))
    base=_brier(np.full_like(actual,float(actual.mean())),actual)
    with np.errstate(invalid="ignore"):
        r=(float(np.corrcoef(pred,actual)[0,1])
           if len(pred)>2 and np.std(pred)>0 and np.std(actual)>0 else np.nan)
    return {
        "brier":brier,
        "brier_skill_vs_floor":((base-brier)/(base-floor)) if base-floor>1e-9 else np.nan,
        "calibration_bias":float(np.mean(pred-actual)),
        "correlation":r,
    }


def bootstrap(pred, actual, *, n=2000, seed=0, alpha=0.05) -> dict:
    """Percentile CIs resampling PAIRS, the experimental unit.

    With 8-20 pairs a point correlation is not reportable on its own; the sprint
    rubric asks for variance, and this is the cheapest honest version. Resampling
    the 12 variants instead would understate uncertainty, because pair-level
    forecast error is the thing being estimated.
    """
    pred,actual=np.asarray(pred,float),np.asarray(actual,float)
    rng=np.random.default_rng(seed)
    draws={k:[] for k in ("brier","brier_skill_vs_floor","calibration_bias","correlation")}
    for _ in range(n):
        idx=rng.integers(0,len(pred),len(pred))
        for k,v in _stats(pred[idx],actual[idx]).items():
            draws[k].append(v)
    out={}
    for k,v in draws.items():
        v=np.asarray(v,float); v=v[np.isfinite(v)]
        out[k]=([float(np.quantile(v,alpha/2)),float(np.quantile(v,1-alpha/2))]
                if len(v)>=n//10 else None)
    return out


def headroom(choices: pd.DataFrame) -> dict:
    """Does realized robustness vary enough for calibration to be measurable?

    The smoke's go/no-go. If every pair comes back 12/12 (or every pair 6/12)
    there is nothing to be calibrated to and the design has no discriminative
    room -- pivot rather than scale up.
    """
    a=choices.groupby("pair_id").chose_canonical.mean()
    return {
        "n_pairs": int(len(a)),
        "actual_mean": float(a.mean()),
        "actual_sd": float(a.std(ddof=1)) if len(a)>1 else 0.0,
        "actual_min": float(a.min()),
        "actual_max": float(a.max()),
        "n_distinct_levels": int(a.nunique()),
        "fraction_at_ceiling": float((a>=1.0).mean()),
        "fraction_at_floor": float((a<=0.5).mean()),
        # Enough spread that a constant forecaster is beatable, and more than
        # two levels so a correlation is not driven by one point.
        "has_discriminative_headroom": bool(a.std(ddof=1)>=0.10 and a.nunique()>=3)
                                        if len(a)>1 else False,
    }


def summarize(forecasts: pd.DataFrame, choices: pd.DataFrame):
    t=item_table(forecasts,choices)
    # The baseline every method must beat: forecast the global mean robustness
    # for every pair. In-sample, so it is generously strong -- beating it is
    # then a real claim rather than an artefact of held-out noise.
    global_mean=float(t.groupby("pair_id").actual_robustness.first().mean())

    result={}
    for method,g in t.groupby("forecast_method"):
        pred=g.predicted_robustness.to_numpy()
        actual=g.actual_robustness.to_numpy()
        brier=_brier(pred,actual)
        base=_brier(np.full_like(actual,global_mean),actual)
        floor=float(np.mean(actual*(1-actual)))
        result[method]={
            "n_pairs":len(g),
            # primary
            "brier":brier,
            "brier_irreducible":floor,
            "brier_baseline_global_mean":base,
            "brier_skill_score":float(1-brier/base) if base>0 else None,
            # Raw skill is capped by the irreducible term -- a perfect forecaster
            # of these fixtures scores ~0.15, which reads as failure. Rescale so
            # 0 is the constant baseline and 1 is the unbeatable floor.
            "brier_skill_vs_floor":(float((base-brier)/(base-floor))
                                    if base-floor>1e-9 else None),
            # calibration
            "calibration_bias":float(np.mean(pred-actual)),
            "calibration_slope":_slope(g.predicted_robustness,g.actual_robustness),
            # discrimination
            "correlation":(float(g.predicted_robustness.corr(g.actual_robustness))
                           if len(g)>2 and g.predicted_robustness.nunique()>1 else None),
            "rank_correlation":(
                float(g.predicted_robustness.corr(g.actual_robustness,method="spearman"))
                if len(g)>2 and g.predicted_robustness.nunique()>1 else None),
            # secondary, kept for continuity with the preregistered gates
            "mae":float(g.abs_error.mean()),
            "rmse":float(np.sqrt(g.sq_error.mean())),
            "mean_signed_error":float(g.error.mean()),
            "predicted_mean":float(g.predicted_robustness.mean()),
            "actual_mean":float(g.actual_robustness.mean()),
            "ci95":bootstrap(pred,actual),
        }
    result["_headroom"]=headroom(choices)
    return result


def demo():
    """A constant forecaster must look good on bias and useless on skill.

    This is the exact failure MAE hid: recentring a badly-ordered forecast.
    """
    rng=np.random.default_rng(0)
    pairs=[f"p{i}" for i in range(20)]
    truth={p:rng.integers(6,13)/N_VARIANTS for p in pairs}
    choices=pd.DataFrame([
        {"pair_id":p,"variant_id":v,
         "chose_canonical":v < round(truth[p]*N_VARIANTS)}
        for p in pairs for v in range(N_VARIANTS)
    ])
    mean=float(np.mean(list(truth.values())))
    fc=pd.DataFrame(
        [{"pair_id":p,"forecast_method":"constant","predicted_robustness":mean}
         for p in pairs]
        +[{"pair_id":p,"forecast_method":"informed",
           "predicted_robustness":min(1.0,truth[p]+0.02)} for p in pairs]
    )
    s=summarize(fc,choices)
    assert abs(s["constant"]["calibration_bias"])<1e-9, s["constant"]
    assert abs(s["constant"]["brier_skill_vs_floor"])<1e-9, s["constant"]
    assert s["constant"]["correlation"] is None
    # Raw skill stays small because the irreducible term dominates; the
    # rescaled one is what shows the forecaster is near-perfect.
    assert s["informed"]["brier_skill_score"]<0.2, s["informed"]
    assert s["informed"]["brier_skill_vs_floor"]>0.95, s["informed"]
    assert s["informed"]["correlation"]>0.95
    assert s["_headroom"]["has_discriminative_headroom"]

    flat=pd.DataFrame([{"pair_id":p,"variant_id":v,"chose_canonical":True}
                       for p in pairs for v in range(N_VARIANTS)])
    assert not headroom(flat)["has_discriminative_headroom"]
    print("ok")


if __name__=="__main__": demo()
