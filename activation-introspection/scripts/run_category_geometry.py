"""Capacity gate for held-out semantic generalization.

Pre-run note: ``notes/23-held-out-semantic-generalization.md``.

``23`` asks whether the model can place an exemplar it has not seen into a
category defined by two others. That is only a question about the model if the
category structure is present at the injection site to begin with. This measures
that, with no generation at all.

Per candidate category pair:

``separation``   reused from ``run_cluster_check`` unchanged, so values stay
comparable to the thresholds in ``16`` and ``19``.

``loo_centroid`` leave-one-out nearest-centroid accuracy over exactly the
four-shot subsets the behavioural run will draw: two demo exemplars per class,
one held-out query exemplar. This is the cheap reader's ceiling on the held-out
task. If it is at chance the task is impossible for a reader *and* a model, and
a behavioural null would say nothing about reporting.

The threshold is **measured, not guessed**. Scrambling the same sixteen vectors
into two arbitrary groups gives a null on real concept vectors -- same norms,
same anisotropy, no category structure -- and a pair passes only by clearing its
99th percentile. An earlier version of this gate used four exemplars per class
and a threshold of 0.75; random directions clear that about five percent of the
time, so it would have passed noise.

    uv run python scripts/run_category_geometry.py
"""

from __future__ import annotations

import itertools
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from run_cluster_check import separation

from introspect import models
from introspect.concepts import ConceptVector, build_concept_vector
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LAYER = 9
OUT = Path("results/category_geometry_v1_summary.json")

#: Eight exemplars per category. Four is far too few: the leave-one-out statistic
#: has a 99th-percentile null of 0.85 at n=4 against 0.66 at n=8, so at n=4 a
#: single lucky pair looks like a category. Eight also leaves room for the
#: held-out query slot to rotate in the behavioural run.
CANDIDATES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "birds_buildings": (
        ("robin", "sparrow", "falcon", "pigeon", "heron", "magpie", "penguin", "owl"),
        ("cathedral", "museum", "warehouse", "stadium",
         "cottage", "factory", "castle", "temple"),
    ),
    "fruit_tools": (
        ("apple", "cherry", "melon", "grape", "peach", "plum", "mango", "lemon"),
        ("wrench", "chisel", "drill", "saw", "pliers", "mallet", "screwdriver", "axe"),
    ),
    "mammals_vehicles": (
        ("rabbit", "otter", "badger", "squirrel", "wolf", "deer", "bear", "fox"),
        ("tractor", "ferry", "glider", "scooter",
         "truck", "submarine", "helicopter", "bicycle"),
    ),
    "body_weather": (
        ("elbow", "thumb", "ankle", "shoulder", "wrist", "knee", "spine", "jaw"),
        ("thunder", "drizzle", "blizzard", "breeze", "fog", "hail", "monsoon", "frost"),
    ),
}

#: Draws for the scrambled null, and the cap on combinations per held-out query.
#: The full enumeration is C(7,2)*C(8,2)=588 per query; sampling 60 keeps the
#: null affordable without shifting it (both are unbiased for the same quantity).
NULL_DRAWS = 200
COMBO_CAP = 60


def loo_centroid(
    a: list[torch.Tensor],
    b: list[torch.Tensor],
    *,
    cap: int = COMBO_CAP,
    seed: int = 0,
) -> float:
    """Fraction of held-out exemplars nearer their own two-shot class centroid.

    Enumerates the splits the behavioural run can draw: one exemplar held out,
    two of its classmates forming its centroid, two from the other class forming
    the rival. Mirrors the four-shot reader exactly.
    """
    rng = random.Random(seed)
    hits = total = 0
    for own, other in ((a, b), (b, a)):
        for i, query in enumerate(own):
            rest = [x for j, x in enumerate(own) if j != i]
            combos = [
                (p, r)
                for p in itertools.combinations(rest, 2)
                for r in itertools.combinations(other, 2)
            ]
            if len(combos) > cap:
                combos = rng.sample(combos, cap)
            for demo_own, demo_other in combos:
                own_sim = torch.dot(query, torch.stack(demo_own).mean(0))
                other_sim = torch.dot(query, torch.stack(demo_other).mean(0))
                hits += int(own_sim > other_sim)
                total += 1
    return hits / total


def scrambled_null(pool: list[torch.Tensor], *, draws: int = NULL_DRAWS) -> list[float]:
    """Regroup the same vectors arbitrarily: real vectors, no category structure."""
    half = len(pool) // 2
    out = []
    for s in range(draws):
        shuffled = list(pool)
        random.Random(s).shuffle(shuffled)
        out.append(loo_centroid(shuffled[:half], shuffled[half:], seed=s))
    return sorted(out)


def _self_check() -> None:
    """Tight separated clusters generalize; interleaved ones must not."""
    a = [torch.tensor([1.0, 0.0]), torch.tensor([0.99, 0.14]),
         torch.tensor([0.98, 0.20]), torch.tensor([1.0, 0.05])]
    b = [torch.tensor([0.0, 1.0]), torch.tensor([0.14, 0.99]),
         torch.tensor([0.20, 0.98]), torch.tensor([0.05, 1.0])]
    assert loo_centroid(a, b) == 1.0, "tight clusters generalize to held-out members"

    x = [a[0], b[0], a[1], b[1]]
    y = [a[2], b[2], a[3], b[3]]
    assert loo_centroid(x, y) < 0.75, "interleaved classes must not generalize"

    # The null must sit at chance on structureless input, or the gate is not a
    # gate. This is the check that caught the n=4 threshold being too loose.
    gen = torch.Generator().manual_seed(0)
    noise = [v / v.norm() for v in torch.randn(16, 256, generator=gen)]
    null = scrambled_null(noise, draws=40)
    assert 0.4 < sum(null) / len(null) < 0.6, "scrambled null must sit at chance"


def main() -> None:
    _self_check()
    preflight_check(MODEL)
    started = time.time()
    model = models.load(MODEL, revision=MODEL_REVISION)

    # Centre on an independent bank. Centring on the evaluation bank's own mean
    # would push the two categories to opposite sides by construction and
    # manufacture exactly the clustering this gate is meant to detect.
    center = torch.stack(
        [build_concept_vector(model, n, LAYER).vector for n in CENTERING_CONCEPTS]
    ).mean(0)

    def units(names: tuple[str, ...]) -> list[torch.Tensor]:
        out = []
        for n in names:
            cv = build_concept_vector(model, n, LAYER)
            centered = ConceptVector(name=n, layer=LAYER, vector=cv.vector - center)
            out.append(centered.unit())
        return out

    rows: dict[str, dict[str, Any]] = {}
    for name, (a_names, b_names) in CANDIDATES.items():
        a, b = units(a_names), units(b_names)
        null = scrambled_null(a + b)
        p99 = null[int(0.99 * len(null)) - 1]
        observed = loo_centroid(a, b)
        rows[name] = {
            "exemplars": {"a": list(a_names), "b": list(b_names)},
            "separation": separation(a, b),
            "loo_centroid": observed,
            "scrambled_null_mean": sum(null) / len(null),
            "scrambled_null_p99": p99,
            "passes": observed > p99,
        }
        print(f"{name:18s} sep={rows[name]['separation']:+.4f} "
              f"loo={observed:.3f} null_p99={p99:.3f} "
              f"{'PASS' if observed > p99 else 'fail'}", flush=True)

    passing = sorted(
        (k for k in rows if rows[k]["passes"]),
        key=lambda k: float(rows[k]["loo_centroid"]),
        reverse=True,
    )
    summary = {
        "what_this_is": (
            "Capacity gate for notes/23. Whether category structure is present "
            "at the injection site at all, before asking whether the model can "
            "report it. No generation."
        ),
        "layer": LAYER,
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "centered_on": list(CENTERING_CONCEPTS),
        "elapsed_seconds": round(time.time() - started, 1),
        "null": {
            "method": (
                "the same sixteen vectors regrouped arbitrarily; real norms and "
                "anisotropy, no category structure"
            ),
            "draws": NULL_DRAWS,
        },
        "candidates": rows,
        "gate": (
            "a pair passes if loo_centroid exceeds the 99th percentile of its own "
            "scrambled null; the two best passing pairs are frozen into the "
            "behavioural run, and the arbitrary arm is matched to their "
            "between-class distance"
        ),
        "passing_ranked": passing,
        "frozen_choice": passing[:2],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\npassing: {passing}")
    print(f"frozen choice for notes/23: {passing[:2]}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
