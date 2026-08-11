"""Exact randomized design for preference foresight.

Each admitted pair has a canonically preferred family `P` and the other `O`. The
model is randomly assigned to actually perform one of them three times, then
makes a binding choice between them.

    realized_change = P(choose P | performed P) - P(choose P | performed O)

Negative is satiation: doing the thing makes you less likely to choose it again.
Positive is momentum. Zero means recent experience does not move the preference.

The arm is crossed with the full Q/K counterbalance, because the whole point of
`choice_prompts` is that an apparent preference shift must not be a label or
order effect that happened to land unevenly across arms.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import random

ARMS = ("performed_preferred", "performed_other")
DOSE = 3
# `blank_reset` and `summary_only` are Branch 18's question, not this one: here
# the model forecast what would happen after it *experienced* the work, so the
# realized arm has to carry that experience.
CONTEXT_MODE = "full_history"
A_LABELS = ("Q", "K")
PRESENTATION_ORDERS = ("QK", "KQ")


# The pair set is a property of the DESIGN, not of whichever model is being run,
# so every model answers the same 10 pairs and the results are comparable. These
# five are the cost-matched arithmetic long band: answer-token ratio 1.00-1.19
# across all pairs, blind-guess baseline 0.03%, and varying `atomic_ops` so the
# preference is about work rather than output length. Character-level families
# are excluded because Qwen3-4B screens 1/16 on them; a frontier model would pass
# them, but then it would not be running the same experiment.
ARITHMETIC_BAND = ("add_ten", "double_numbers", "running_totals",
                   "sort_numbers", "sort_numbers_desc")


@dataclass(frozen=True)
class Cell:
    pair_id: str
    arm: str
    a_label: str
    presentation_order: str
    replicate: int


def cells(pair_ids, replicates: int = 1, seed: int = 20260810):
    out = [
        Cell(pair, arm, lab, order, r)
        for pair, arm, lab, order, r in itertools.product(
            pair_ids, ARMS, A_LABELS, PRESENTATION_ORDERS, range(replicates))
    ]
    # Shuffled so that if anything drifts over the course of a run -- a provider
    # switch, a rate-limit backoff, context creeping -- it does not land on one
    # arm.
    random.Random(seed).shuffle(out)
    return out


def cells_per_pair(replicates: int = 1) -> int:
    return len(ARMS) * len(A_LABELS) * len(PRESENTATION_ORDERS) * replicates


def calls_per_pair(replicates: int = 1) -> int:
    """Admission + both forecasts + treatment, for budgeting before spending."""
    admission = 4 * 2                      # 4 counterbalanced choices, each binding
    forecast = 2 * 2                       # 2 counterfactuals, up to 2 turns each
    per_cell = DOSE + 2                    # 3 treatment tasks, choice, post-task
    return admission + forecast + cells_per_pair(replicates) * per_cell


def validate(pair_ids, replicates: int = 1):
    cs = cells(pair_ids, replicates)
    assert len(cs) == len(pair_ids) * cells_per_pair(replicates)
    for pair in pair_ids:
        sub = [c for c in cs if c.pair_id == pair]
        for arm in ARMS:
            # Each arm must carry the identical counterbalance, or an arm
            # difference is confounded with a label/order difference.
            got = sorted((c.a_label, c.presentation_order)
                         for c in sub if c.arm == arm)
            want = sorted(itertools.chain.from_iterable(
                [list(itertools.product(A_LABELS, PRESENTATION_ORDERS))] * replicates))
            assert got == want, (pair, arm, got)
    return {"pairs": len(pair_ids), "cells": len(cs),
            "cells_per_pair": cells_per_pair(replicates),
            "calls_per_pair": calls_per_pair(replicates)}


if __name__ == "__main__":
    print(validate([f"p{i}" for i in range(3)], replicates=2))
