"""Bank audit: does the fitted reader point along every held-out direction?

Pre-run note: ``notes/13-shared-axis-audit.md``, written before this ran.

``notes/12`` found a straight-line reader transferring perfectly to eight concept
directions it never saw. That is only possible if the directions share a common
component. This measures whether they do. It trains nothing, scores no reporting
task, and makes no new claim about the model -- it is an audit of this
repository's own concept bank.

State capture and reader fitting are imported from ``run_trained_vs_probe`` rather
than reimplemented, so the geometry measured here belongs to the published run and
not to a lookalike.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from introspect import models  # noqa: E402
from introspect.concepts import ConceptVector, build_bank, pairwise_cosines  # noqa: E402
from introspect.preflight import check as preflight_check  # noqa: E402
from introspect.report_training import (  # noqa: E402
    CENTERING_CONCEPTS,
    EVAL_CONCEPTS,
    TRAIN_CARRIERS,
    TRAIN_CONCEPTS,
)
from run_trained_vs_probe import (  # noqa: E402
    LAYER,
    MODEL,
    MODEL_REVISION,
    STRENGTH,
    _fit_logistic,
    _state,
)

OUT = Path("results/bank_audit_v1_summary.json")


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(_unit(a), _unit(b)))


def _spread(values: list[float]) -> dict[str, float]:
    t = torch.tensor(values)
    return {
        "min": float(t.min()),
        "max": float(t.max()),
        "mean": float(t.mean()),
        "mean_abs": float(t.abs().mean()),
        "n_positive": int((t > 0).sum()),
        "n": len(values),
    }


def _self_check() -> None:
    """The arithmetic, on vectors whose answers are known."""
    v = torch.randn(64)
    assert abs(_cos(v, v) - 1.0) < 1e-5
    assert abs(_cos(v, -v) + 1.0) < 1e-5
    orthogonal = torch.zeros(64)
    orthogonal[0] = 1.0
    other = torch.zeros(64)
    other[1] = 1.0
    assert abs(_cos(orthogonal, other)) < 1e-6
    # n unit vectors spread evenly average to about 1/sqrt(n); this is the null
    # every "shared axis" number below is compared against.
    sample = torch.stack([_unit(torch.randn(2048)) for _ in range(8)]).mean(0)
    assert 0.2 < float(sample.norm()) * (8**0.5) < 2.0


def run() -> None:
    _self_check()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {OUT}")
    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

        # Banks, built exactly as the published run builds them.
        centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
        center = torch.stack([cv.vector for cv in centering.values()]).mean(dim=0)

        def centered(names: tuple[str, ...]) -> dict[str, ConceptVector]:
            raw = build_bank(model, LAYER, list(names), center=False)
            return {
                name: ConceptVector(name=name, layer=LAYER, vector=cv.vector - center)
                for name, cv in raw.items()
            }

        train_bank = centered(TRAIN_CONCEPTS)
        eval_bank = centered(EVAL_CONCEPTS)
        print("banks built", flush=True)

        # The same 96 training states the readers were fitted on.
        states, signs_list = [], []
        for concept in TRAIN_CONCEPTS:
            for carrier in TRAIN_CARRIERS:
                for sign in (1, -1):
                    states.append(_state(model, carrier, train_bank[concept], sign))
                    signs_list.append(sign)
            print(f"train states: {concept}", flush=True)
        features = torch.stack(states)
        signs = torch.tensor(signs_list, dtype=torch.float32)

        centroid_weight = features[signs > 0].mean(0) - features[signs < 0].mean(0)
        logistic_weight = _fit_logistic(features, signs)[:-1]  # drop the bias term

        # The shared-axis magnitude. Evenly spread directions average to about
        # 1/sqrt(n); a value far above that is a common component.
        train_units = torch.stack([_unit(cv.vector) for cv in train_bank.values()])
        eval_units = torch.stack([_unit(cv.vector) for cv in eval_bank.values()])
        n = len(train_units)
        even_spread_null = n**-0.5

        # The decisive measurement: does each reader point along every unseen
        # direction? Chance for one pair in d dimensions is about 1/sqrt(d).
        d = int(features.shape[1])
        per_direction = {
            name: {
                "centroid": _cos(centroid_weight, eval_bank[name].vector),
                "logistic": _cos(logistic_weight, eval_bank[name].vector),
            }
            for name in eval_bank
        }

        summary = {
            "what_this_is": (
                "Geometry audit of the concept banks behind notes/12. No training, "
                "no reporting task, no new claim about the model."
            ),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "strength": STRENGTH,
            "d_model": d,
            "n_train_states": len(states),
            "chance_cosine_one_pair": d**-0.5,
            "evenly_spread_null_for_mean_direction": even_spread_null,
            "shared_axis": {
                "train_mean_unit_norm": float(train_units.mean(0).norm()),
                "eval_mean_unit_norm": float(eval_units.mean(0).norm()),
                "cos_train_mean_to_eval_mean": _cos(train_units.mean(0), eval_units.mean(0)),
                "cos_centroid_weight_to_train_mean": _cos(centroid_weight, train_units.mean(0)),
            },
            "within_bank_overlap": {
                "train": _spread([abs(v) for v in pairwise_cosines(train_bank).values()]),
                "eval": _spread([abs(v) for v in pairwise_cosines(eval_bank).values()]),
            },
            "reader_to_heldout_direction": {
                "centroid": _spread([v["centroid"] for v in per_direction.values()]),
                "logistic": _spread([v["logistic"] for v in per_direction.values()]),
                "per_direction": per_direction,
            },
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"wrote {OUT}", flush=True)
    finally:
        model.free()


if __name__ == "__main__":
    run()
