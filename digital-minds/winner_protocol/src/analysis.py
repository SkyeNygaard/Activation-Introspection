"""Carrier-clustered analysis for the proposed confirmation."""
from __future__ import annotations
from collections import defaultdict
import numpy as np

def oriented_p_correct(row: dict, arm: str) -> float:
    return float(row["arms"][arm]["p_correct"])

def carrier_means(rows: list[dict], arm: str, persona: str) -> dict[str, float]:
    by = defaultdict(list)
    for row in rows:
        if row["persona"] == persona:
            by[str(row["carrier"])].append(oriented_p_correct(row, arm))
    return {carrier: float(np.mean(vals)) for carrier, vals in by.items()}

def carrier_contrast(rows: list[dict], persona: str) -> dict[str, float]:
    t = carrier_means(rows, "target", persona)
    q = carrier_means(rows, "query_only", persona)
    if set(t) != set(q):
        raise ValueError("carrier grid differs between target/query-only")
    return {c: t[c] - q[c] for c in t}

def bootstrap_mean(values, *, seed=0, n_boot=20000):
    x = np.asarray(list(values), dtype=float)
    if len(x) < 2:
        raise ValueError("need >=2 carrier units")
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return {
        "value": float(x.mean()),
        "ci95": [float(np.quantile(draws,.025)), float(np.quantile(draws,.975))],
        "n_carriers": len(x),
    }

def primary_summary(rows: list[dict]) -> dict:
    out = {}
    for i, persona in enumerate(("neutral","upbeat","downbeat")):
        delta = carrier_contrast(rows, persona)
        out[persona] = bootstrap_mean(delta.values(), seed=20260810+i)
        out[persona]["gate_lower_gt_0.10"] = out[persona]["ci95"][0] > .10
    out["passed_all_personas"] = all(v["gate_lower_gt_0.10"] for v in out.values())
    return out


def semantic_carrier_state_effect(rows: list[dict], persona: str) -> dict[str, float]:
    """Mean rating(+state)-rating(-state), nuisance-averaged within carrier."""
    by = defaultdict(lambda: {+1: [], -1: []})
    for row in rows:
        if row["persona"] != persona or "semantic_rating" not in row:
            continue
        by[str(row["carrier"])][int(row["query_sign"])].append(
            float(row["semantic_rating"])
        )
    out = {}
    for carrier, states in by.items():
        if not states[+1] or not states[-1]:
            raise ValueError(f"carrier {carrier} lacks +/- semantic twins")
        out[carrier] = float(np.mean(states[+1]) - np.mean(states[-1]))
    return out

def semantic_persona_offset(rows: list[dict]) -> dict[str, float]:
    """Persona offset after averaging over hidden sign and nuisance cells."""
    by = defaultdict(list)
    for row in rows:
        if "semantic_rating" in row:
            by[(str(row["carrier"]), row["persona"])].append(
                float(row["semantic_rating"])
            )
    carriers = sorted({k[0] for k in by})
    return {
        carrier: float(np.mean(by[(carrier,"upbeat")]) - np.mean(by[(carrier,"downbeat")]))
        for carrier in carriers
    }

def semantic_summary(rows: list[dict]) -> dict:
    out = {"state_effect": {}}
    for i, persona in enumerate(("neutral","upbeat","downbeat")):
        vals = semantic_carrier_state_effect(rows, persona)
        out["state_effect"][persona] = bootstrap_mean(
            vals.values(), seed=20260820+i
        )
    offsets = semantic_persona_offset(rows)
    out["upbeat_minus_downbeat_offset"] = bootstrap_mean(
        offsets.values(), seed=20260830
    )
    return out
