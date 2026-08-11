"""Does the model know how its own preference will change?

Three questions, scored separately because they can come apart:

1. **causal effect** -- does performing a task change the next binding choice?
2. **foresight** -- does the model's prospective forecast track the change it
   actually undergoes?
3. **incremental value** -- does asking the model beat predicting it from its own
   observable prior behaviour?

(3) is the one that makes this about self-knowledge rather than about a model
being able to recite a plausible psychology fact. A forecast that merely says
"repetition causes satiation" for every pair scores well on (2) and adds nothing
over the constant baseline, and only (3) exposes that.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def boot_ci(fn, *arrays, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI for `fn(*arrays)`, resampling PAIRS with replacement.

    The pair is the experimental unit: resampling individual binding trials would
    treat the four counterbalanced cells of one pair as four independent pairs
    and understate the interval badly at n=10.
    """
    arrays = [np.asarray(a, float) for a in arrays]
    k = len(arrays[0])
    if k < 3:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, k, k)
        try:
            v = fn(*[a[idx] for a in arrays])
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(v):
            draws.append(float(v))
    if len(draws) < n // 10:
        return None
    return [float(np.quantile(draws, alpha / 2)),
            float(np.quantile(draws, 1 - alpha / 2))]


def _corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def pair_table(choices: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """One row per pair: predicted change, realized change, and covariates."""
    req = {"pair_id", "arm", "chose_preferred"}
    if not req <= set(choices):
        raise ValueError(req - set(choices))
    freq = {"pair_id", "forecast_after_preferred", "forecast_after_other"}
    if not freq <= set(forecasts):
        raise ValueError(freq - set(forecasts))

    rows = []
    for pair, g in choices.groupby("pair_id"):
        after_p = g[g.arm == "performed_preferred"].chose_preferred.astype(float)
        after_o = g[g.arm == "performed_other"].chose_preferred.astype(float)
        if after_p.empty or after_o.empty:
            raise ValueError(f"{pair} is missing a randomized arm")
        rows.append({
            "pair_id": pair,
            "n_after_preferred": len(after_p),
            "n_after_other": len(after_o),
            "realized_after_preferred": float(after_p.mean()),
            "realized_after_other": float(after_o.mean()),
            "realized_change": float(after_p.mean() - after_o.mean()),
        })
    t = pd.DataFrame(rows).merge(forecasts, on="pair_id", validate="one_to_one")
    t["predicted_change"] = (t.forecast_after_preferred - t.forecast_after_other)
    t["error"] = t.predicted_change - t.realized_change
    return t


def summarize(choices: pd.DataFrame, forecasts: pd.DataFrame,
              baseline_strength: pd.DataFrame | None = None) -> dict:
    t = pair_table(choices, forecasts)
    real = t.realized_change.to_numpy(float)
    pred = t.predicted_change.to_numpy(float)
    n = len(t)

    out = {"n_pairs": n, "pairs": t.round(4).to_dict("records")}

    # 1. Does experience move the preference at all?
    out["causal_effect"] = {
        "mean_realized_change": float(real.mean()),
        "sd_across_pairs": float(real.std(ddof=1)) if n > 1 else None,
        "n_pairs_negative": int((real < 0).sum()),
        "n_pairs_positive": int((real > 0).sum()),
        "n_pairs_zero": int((real == 0).sum()),
        "ci95_mean": boot_ci(lambda x: x.mean(), real),
        # Without spread across pairs there is nothing for a forecast to track,
        # however large the average effect is.
        "has_variation_to_forecast": bool(n > 1 and real.std(ddof=1) >= 0.10),
    }

    # 2. Does the model's forecast track it?
    mse = float(np.mean((pred - real) ** 2))
    const_mse = float(np.mean((real.mean() - real) ** 2))
    out["foresight"] = {
        "mean_predicted_change": float(pred.mean()),
        "mse": mse,
        "bias": float(np.mean(pred - real)),
        "correlation": _corr(pred, real),
        "rank_correlation": (float(t.predicted_change.corr(
            t.realized_change, method="spearman"))
            if n > 2 and pred.std() > 0 and real.std() > 0 else None),
        "sign_agreement": float(np.mean(np.sign(pred) == np.sign(real))),
        "ci95_correlation": boot_ci(_corr, pred, real),
        "ci95_bias": boot_ci(lambda p, r: np.mean(p - r), pred, real),
    }

    # 3. Does it beat the outside view?
    baselines = {"constant_mean_change": const_mse}
    if baseline_strength is not None:
        b = t.merge(baseline_strength, on="pair_id", validate="one_to_one")
        # The outside view available without asking the model anything: how
        # stable its preference already looked in the admission screen.
        if b.agreement.nunique() > 1 and n > 2:
            slope, intercept = np.polyfit(b.agreement, b.realized_change, 1)
            fitted = slope * b.agreement + intercept
            baselines["admission_strength_insample"] = float(
                np.mean((fitted - b.realized_change) ** 2))
            baselines["_note_admission"] = (
                "fitted in-sample on the same pairs, so it is generously strong; "
                "the model's forecast has to beat a baseline that saw the answers")
    out["baselines"] = baselines
    out["skill_vs_constant"] = (float(1 - mse / const_mse)
                                if const_mse > 1e-12 else None)
    return out


def demo():
    """Recover an injected effect, and catch the forecast that only sounds right.

    Two synthetic models over the same realized data: one whose forecast tracks
    each pair, one that predicts the same textbook satiation everywhere. Both
    have small bias. Only the skill score separates them.
    """
    rng = np.random.default_rng(0)
    pairs = [f"p{i}" for i in range(12)]
    # Centred on 0.5 in the other arm, so a POSITIVE change is representable.
    # Pinning the other arm at 1.0 silently truncated every positive effect to
    # zero and the fixture could then only ever show satiation.
    true_change = {p: float(rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5]))
                   for p in pairs}
    OTHER = 0.5

    rows = []
    for p in pairs:
        n_keep = int(round((OTHER + true_change[p]) * 4))
        for i in range(4):
            rows.append({"pair_id": p, "arm": "performed_preferred",
                         "chose_preferred": i < n_keep})
            rows.append({"pair_id": p, "arm": "performed_other",
                         "chose_preferred": i < int(OTHER * 4)})
    choices = pd.DataFrame(rows)

    informed = pd.DataFrame([
        {"pair_id": p, "forecast_after_preferred": OTHER + true_change[p],
         "forecast_after_other": OTHER} for p in pairs])
    # Recites the average effect for every pair: unbiased by construction, and
    # exactly as good as the constant baseline.
    mean_change = float(np.mean(list(true_change.values())))
    generic = pd.DataFrame([
        {"pair_id": p, "forecast_after_preferred": OTHER + mean_change,
         "forecast_after_other": OTHER} for p in pairs])

    si = summarize(choices, informed)
    sg = summarize(choices, generic)

    for s in (si, sg):
        got = {r["pair_id"]: r["realized_change"] for r in s["pairs"]}
        assert all(abs(got[p] - true_change[p]) < 1e-9 for p in pairs), got

    assert si["foresight"]["correlation"] > 0.99, si["foresight"]
    assert si["skill_vs_constant"] > 0.99, si["skill_vs_constant"]
    # The generic forecaster is near-unbiased and still worthless: it cannot
    # distinguish the pairs. This is the comparison MAE-style scoring misses.
    assert abs(sg["foresight"]["bias"]) < 0.03, sg["foresight"]
    assert sg["skill_vs_constant"] < 0.05, sg["skill_vs_constant"]
    assert si["causal_effect"]["has_variation_to_forecast"]

    flat = pd.DataFrame([
        {"pair_id": p, "arm": a, "chose_preferred": a == "performed_other"}
        for p in pairs for a in ("performed_preferred", "performed_other")])
    assert not summarize(flat, informed)["causal_effect"][
        "has_variation_to_forecast"]
    print("ok")


if __name__ == "__main__":
    demo()
