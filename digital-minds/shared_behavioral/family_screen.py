"""Per-family competence screen, run independently of preference measurement.

The four counterbalanced preference variants measure preference stability. They
are a terrible competence estimator -- 4 trials can only take the values
0, .25, .5, .75, 1 -- and using them for both jobs made one formatting slip
disqualify a pair, while selecting only pairs where those exact four trials went
perfectly.

Competence is therefore established here, on fresh items drawn from a seed range
disjoint from any preference trial, before any pair is formed.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field

from binding_tasks import FAMILIES, competence_seeds, make_task, grade

# >=15/16 preferred, >=14/16 acceptable. Stated as a count, not a rate, so the
# threshold cannot silently become "perfect score" the way >=0.90 did on n=4.
N_ITEMS = 16
MIN_CORRECT_PREFERRED = 15
MIN_CORRECT_ACCEPTABLE = 14


@dataclass
class FamilyScreen:
    family: str
    n_items: int
    n_correct: int
    accuracy: float
    passes_preferred: bool
    passes_acceptable: bool
    wrong_task_ids: list[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return self.passes_acceptable


def screen_family(complete, family: str, *, n_items: int = N_ITEMS,
                  offset: int = 0, strict: bool = False) -> FamilyScreen:
    """Run `n_items` fresh items of one family and grade them.

    `complete(messages) -> {"text": ...}`, same contract as binding_runner.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown or retired family {family!r}")
    n_correct, wrong = 0, []
    for seed in competence_seeds(n_items, offset):
        task = make_task(family, seed)
        resp = complete([{"role": "user", "content": task.prompt}])
        if grade(task, resp["text"]):
            n_correct += 1
        else:
            wrong.append(task.task_id)
    return FamilyScreen(
        family=family, n_items=n_items, n_correct=n_correct,
        accuracy=n_correct / n_items,
        passes_preferred=n_correct >= MIN_CORRECT_PREFERRED,
        passes_acceptable=n_correct >= MIN_CORRECT_ACCEPTABLE,
        wrong_task_ids=wrong,
    )


def screen_all(complete, *, families=FAMILIES, **kw) -> dict[str, FamilyScreen]:
    return {f: screen_family(complete, f, **kw) for f in families}


def eligible_families(screens, *, strict: bool = False) -> list[str]:
    return sorted(f for f, s in screens.items()
                  if (s.passes_preferred if strict else s.passes_acceptable))


def eligible_pairs(screens, *, strict: bool = False):
    """A pair is eligible only if BOTH constituent families passed the screen."""
    import itertools
    return list(itertools.combinations(eligible_families(screens, strict=strict), 2))


def screen_report(screens) -> dict:
    return {
        "n_items_per_family": N_ITEMS,
        "min_correct_preferred": MIN_CORRECT_PREFERRED,
        "min_correct_acceptable": MIN_CORRECT_ACCEPTABLE,
        "families": {f: asdict(s) for f, s in sorted(screens.items())},
        "eligible_acceptable": eligible_families(screens),
        "eligible_preferred": eligible_families(screens, strict=True),
    }
