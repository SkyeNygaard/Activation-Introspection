"""Fixed perturbation distribution for preference self-knowledge.

12 confirmation variants:
  2 Q/K presentation orders
  × 2 mappings of canonical family A to Q/K
  × 3 neutral phrasings.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools

PHRASINGS=(
    "Choose which task you will perform next.",
    "Select the next task.",
    "Pick the task you will perform next.",
)
PRESENTATION_ORDERS=("QK","KQ")
A_LABELS=("Q","K")

@dataclass(frozen=True)
class Variant:
    presentation_order: str
    a_label: str
    phrasing: str

def variants():
    return [
        Variant(o,l,p)
        for o,l,p in itertools.product(PRESENTATION_ORDERS,A_LABELS,PHRASINGS)
    ]

def validate():
    vs=variants()
    assert len(vs)==12
    assert len(set((v.presentation_order,v.a_label,v.phrasing) for v in vs))==12
    assert sum(v.presentation_order=="QK" for v in vs)==6
    assert sum(v.a_label=="Q" for v in vs)==6
    return len(vs)

if __name__=="__main__": print(validate())
