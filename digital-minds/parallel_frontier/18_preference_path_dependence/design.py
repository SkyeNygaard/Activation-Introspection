"""Exact randomized design for preference path-dependence experiment."""
from __future__ import annotations
from dataclasses import dataclass
import random

ASSIGNMENTS=("A","B")
DOSES=(1,3)
CONTEXT_MODES=("full_history","summary_only","blank_reset")

@dataclass(frozen=True)
class Cell:
    pair_id: str
    assignment: str
    dose: int
    context_mode: str
    replicate: int

def exact_cells(pair_ids, replicates=2, seed=20260810):
    cells=[
        Cell(pair,a,d,c,r)
        for pair in pair_ids
        for a in ASSIGNMENTS
        for d in DOSES
        for c in CONTEXT_MODES
        for r in range(replicates)
    ]
    rng=random.Random(seed)
    rng.shuffle(cells)
    return cells

def validate(cells,pair_ids,replicates=2):
    expected=len(pair_ids)*2*2*3*replicates
    assert len(cells)==expected
    for pair in pair_ids:
        sub=[x for x in cells if x.pair_id==pair]
        for a in ASSIGNMENTS:
            for d in DOSES:
                for c in CONTEXT_MODES:
                    assert sum(
                        x.assignment==a and x.dose==d and x.context_mode==c
                        for x in sub
                    )==replicates
    return {"cells":len(cells),"pairs":len(pair_ids)}
