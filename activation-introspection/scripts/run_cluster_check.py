"""Does representational clustering predict which rules the interface can learn?

Pre-run note: the second section of ``notes/16-visible-rule-capacity.md``.

``16`` found the four-shot interface learns `category` (0.885) and fails
`first_letter` (0.479) and `parity` (0.469), and inferred that it matches on
representational similarity rather than inducing rules. This measures that
directly: for each rule, how much closer are items to their own class than to the
other one?

Prompting-free. Rule definitions are imported from the screen, not restated.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from introspect import models
from introspect.hooks import capture
from introspect.preflight import check as preflight_check
from run_visible_rule_screen import MODEL, MODEL_REVISION, RULES

LAYERS = (9, 18, 27)
OUT = Path("results/cluster_check_v1_summary.json")

#: From notes/16, so the two are compared without re-running the screen.
ACCURACY = {
    "lexical_seen": 0.979,
    "category": 0.885,
    "magnitude": 0.729,
    "first_letter": 0.479,
    "lexical_unseen": 0.490,
    "parity": 0.469,
}


@torch.no_grad()
def embed(model: models.LoadedModel, item: str, layer: int) -> torch.Tensor:
    """Last-token state in the same frame the prompt uses."""
    ids = model.encode(f"Item: {item}")
    with capture(model, [layer]) as store:
        model.model(ids)
    v = store.last_token(layer)[0].float()
    return v / (v.norm() + 1e-8)


def separation(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    """Mean within-class similarity minus mean between-class similarity.

    Zero means the two classes are no tighter internally than they are to each
    other -- no cluster for a query to fall into.
    """
    def mean_pairs(xs: list[torch.Tensor], ys: list[torch.Tensor], same: bool) -> float:
        vals = [
            float(torch.dot(x, y))
            for i, x in enumerate(xs)
            for j, y in enumerate(ys)
            if not (same and i >= j)
        ]
        return sum(vals) / len(vals) if vals else 0.0

    within = (mean_pairs(a, a, True) + mean_pairs(b, b, True)) / 2
    return within - mean_pairs(a, b, False)


def _self_check() -> None:
    tight_a = [torch.tensor([1.0, 0.0]), torch.tensor([0.99, 0.14])]
    tight_b = [torch.tensor([0.0, 1.0]), torch.tensor([0.14, 0.99])]
    assert separation(tight_a, tight_b) > 0.5, "distinct tight clusters separate"

    # No cluster structure: mutually orthogonal members, so within and between
    # similarity are both zero. (An earlier version of this check used vectors
    # that were *anti*-clustered -- each item nearer a member of the other class
    # than its own classmate -- which gives a large negative separation, not zero.)
    basis = torch.eye(4)
    assert abs(separation([basis[0], basis[1]], [basis[2], basis[3]])) < 1e-6

    # And the anti-clustered case really is strongly negative, which is a signal
    # in its own right rather than an absence of one.
    assert separation([basis[0], basis[1]], [basis[0], basis[1]]) < -0.3


def run() -> None:
    _self_check()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {OUT}")
    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

        by_rule: dict[str, dict[str, float]] = {}
        for rule, build in RULES.items():
            per_layer: dict[str, float] = {}
            for layer in LAYERS:
                scores = []
                for class_a, class_b, query_a, query_b in build():
                    # Include the query items: they are what must land in a class.
                    a = [embed(model, x, layer) for x in [*class_a, query_a]]
                    b = [embed(model, x, layer) for x in [*class_b, query_b]]
                    scores.append(separation(a, b))
                per_layer[f"layer_{layer}"] = sum(scores) / len(scores)
            by_rule[rule] = {**per_layer, "accuracy_from_notes_16": ACCURACY[rule]}
            print(f"{rule}: {by_rule[rule]}", flush=True)

        # Does separation order the rules the way accuracy does?
        real = [r for r in RULES if r != "lexical_seen"]  # seen query is not induction
        ranking = {
            f"layer_{layer}": {
                "by_separation": sorted(
                    real, key=lambda r: by_rule[r][f"layer_{layer}"], reverse=True
                ),
                "by_accuracy": sorted(real, key=lambda r: ACCURACY[r], reverse=True),
            }
            for layer in LAYERS
        }
        summary = {
            "what_this_is": (
                "Within-class minus between-class similarity per rule, against the "
                "accuracies from notes/16. Measures the cluster account directly."
            ),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layers": list(LAYERS),
            "by_rule": by_rule,
            "ranking": ranking,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"wrote {OUT}", flush=True)
    finally:
        model.free()


if __name__ == "__main__":
    run()
