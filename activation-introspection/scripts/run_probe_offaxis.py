"""Does the fixed probe from notes/12 survive directions off the shared axis?

Pre-run note: ``notes/13-shared-axis-audit.md``, written before this ran.

The bank audit showed the reader in ``notes/12`` is the average concept direction,
which points positively along every held-out concept direction. That predicts the
same reader collapses to chance on magnitude-matched *random* directions, where
``notes/08`` puts the trained adapter at 0.913-0.955. This runs that comparison.

Readers are fitted exactly as ``run_trained_vs_probe.py`` fits them, on the same
training bank, and then scored on random controls instead of concept directions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from introspect import models  # noqa: E402
from introspect.concepts import ConceptVector, build_bank, random_control  # noqa: E402
from introspect.preflight import check as preflight_check  # noqa: E402
from introspect.report_training import (  # noqa: E402
    CENTERING_CONCEPTS,
    EVAL_CARRIERS,
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

OUT = Path("results/probe_offaxis_v1_summary.json")

#: notes/08, trained adapters on magnitude-matched random directions.
TRAINED_ON_RANDOM = (0.955, 0.913)
#: notes/08, untrained model on the same controls.
UNTRAINED_ON_RANDOM = 0.513


def _twin_pair(correct: dict[tuple[str, str, int], bool]) -> float:
    cells = {(name, carrier) for name, carrier, _ in correct}
    return sum(
        correct[(name, carrier, 1)] and correct[(name, carrier, -1)] for name, carrier in cells
    ) / len(cells)


def _self_check() -> None:
    """A cell counts only when both signs are right."""
    both = {("a", "c", 1): True, ("a", "c", -1): True}
    one = {("a", "c", 1): True, ("a", "c", -1): False}
    assert _twin_pair(both) == 1.0
    assert _twin_pair(one) == 0.0


def run() -> None:
    _self_check()
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {OUT}")
    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

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

        # Readers, fitted exactly as notes/12 fits them.
        states, signs_list = [], []
        for concept in TRAIN_CONCEPTS:
            for carrier in TRAIN_CARRIERS:
                for sign in (1, -1):
                    states.append(_state(model, carrier, train_bank[concept], sign))
                    signs_list.append(sign)
            print(f"train states: {concept}", flush=True)
        features = torch.stack(states)
        signs = torch.tensor(signs_list, dtype=torch.float32)
        positive = features[signs > 0].mean(0)
        negative = features[signs < 0].mean(0)
        logistic = _fit_logistic(features, signs)
        weight, bias = logistic[:-1], logistic[-1]

        # One magnitude-matched random direction per held-out concept, built by
        # the apparatus's own control constructor so the norm matches.
        arms = {
            "concept": eval_bank,
            "random": {
                f"random[{name}]": random_control(eval_bank[name], seed=0) for name in eval_bank
            },
        }

        results: dict[str, dict[str, float]] = {}
        per_direction: dict[str, dict[str, float]] = {}
        for arm, bank in arms.items():
            correct: dict[str, dict[tuple[str, str, int], bool]] = {"centroid": {}, "logistic": {}}
            hits: dict[str, dict[str, int]] = {}
            for name, direction in bank.items():
                hits[name] = {"centroid": 0, "logistic": 0, "n": 0}
                for carrier in EVAL_CARRIERS:
                    for sign in (1, -1):
                        state = _state(model, carrier, direction, sign)
                        c_pred = (
                            1 if (state - positive).norm() <= (state - negative).norm() else -1
                        )
                        l_pred = 1 if float(torch.dot(state, weight) + bias) > 0 else -1
                        correct["centroid"][(name, carrier, sign)] = c_pred == sign
                        correct["logistic"][(name, carrier, sign)] = l_pred == sign
                        hits[name]["centroid"] += int(c_pred == sign)
                        hits[name]["logistic"] += int(l_pred == sign)
                        hits[name]["n"] += 1
                print(f"{arm}: {name}", flush=True)
            results[arm] = {
                "centroid_row": sum(correct["centroid"].values()) / len(correct["centroid"]),
                "logistic_row": sum(correct["logistic"].values()) / len(correct["logistic"]),
                "centroid_twin_pair": _twin_pair(correct["centroid"]),
                "logistic_twin_pair": _twin_pair(correct["logistic"]),
            }
            per_direction[arm] = {
                name: {k: v / h["n"] for k, v in h.items() if k != "n"}
                for name, h in hits.items()
            }

        summary = {
            "what_this_is": (
                "The fixed readers from notes/12, scored on magnitude-matched random "
                "directions as well as concept directions. No training."
            ),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "strength": STRENGTH,
            "arms": results,
            "per_direction_row_accuracy": per_direction,
            "reference_from_notes_08": {
                "trained_adapters_on_random": list(TRAINED_ON_RANDOM),
                "untrained_model_on_random": UNTRAINED_ON_RANDOM,
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
