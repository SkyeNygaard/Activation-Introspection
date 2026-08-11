"""Model-free falsifiers.

This does not predict Qwen performance. It tests whether the DESIGN statistic can
be faked by simple shortcut families.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

AGENTS = (
    "true_demo_reader",
    "fixed_label_bias",
    "query_sign_bias",
    "persona_bias",
    "mapping_only_leak",
    "damage_only",
)

def run(seed=20260810, n_carriers=24):
    rng = np.random.default_rng(seed)
    rows = []
    for agent in AGENTS:
        for carrier in range(n_carriers):
            carrier_noise = rng.normal(0,.15)
            for persona_i, persona in enumerate(("neutral","upbeat","downbeat")):
                for mapping in (0,1):
                    for state in (-1,+1):
                        # correct orientation: +1 score means correct opaque label
                        correct_sign = +1 if ((state == +1) == (mapping == 0)) else -1
                        for arm in ("target","query_only"):
                            q_score = rng.normal(0,.35)
                            if agent == "true_demo_reader" and arm == "target":
                                q_score += 1.1 * correct_sign + carrier_noise
                            elif agent == "fixed_label_bias":
                                q_score += 1.0
                            elif agent == "query_sign_bias":
                                q_score += 1.0 * state
                            elif agent == "persona_bias":
                                q_score += (persona_i - 1) * .9
                            elif agent == "mapping_only_leak":
                                q_score += (1 if mapping == 0 else -1) * .9
                            elif agent == "damage_only":
                                q_score += .8
                            p_q = 1/(1+np.exp(-q_score))
                            p_correct = p_q if correct_sign == +1 else 1-p_q
                            rows.append({
                                "agent":agent, "carrier":carrier, "persona":persona,
                                "mapping":mapping, "state":state, "arm":arm,
                                "p_correct":p_correct,
                            })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = run()
    cell = df.groupby(["agent","carrier","persona","arm"])["p_correct"].mean().unstack("arm")
    cell["target_minus_query_only"] = cell["target"] - cell["query_only"]
    summary = (
        cell.groupby(["agent","persona"])["target_minus_query_only"]
        .agg(["mean","std"]).reset_index()
    )
    print(summary.round(3).to_string(index=False))
    Path("simulations").mkdir(exist_ok=True)
    summary.to_csv("simulations/shortcut_screen.csv", index=False)
