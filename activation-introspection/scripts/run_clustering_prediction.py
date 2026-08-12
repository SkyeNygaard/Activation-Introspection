"""Does class clustering predict which hidden rules the interface can learn?

Pre-run note: ``notes/19-clustering-predicts-learnability.md``.

notes/16 found the relationship on six rules, scored after the fact. This makes it
a prediction. Phase ``measure`` computes class separation for fourteen fresh rules
and writes a protocol with a pass/fail prediction for each, from separation alone.
Phase ``test`` refuses to run without that protocol, then scores the predictions.

Thresholds are fixed from notes/16's gaps and are not tunable here.

    uv run python scripts/run_clustering_prediction.py --phase measure
    uv run python scripts/run_clustering_prediction.py --phase test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from introspect import models
from introspect.preflight import check as preflight_check
from run_cluster_check import LAYERS, embed, separation
from run_visible_rule_screen import (
    ANSWER_PREFIX,
    LABELS,
    MODEL,
    MODEL_REVISION,
    cells,
    render,
    score,
    twin_pair,
)

PROTOCOL = Path("results/clustering_prediction_protocol_v1.json")
OUT = Path("results/clustering_prediction_v1_raw.jsonl")

#: Midpoints of the empty gaps in notes/16. Frozen; phase `test` may not change them.
SEPARATION_THRESHOLD = 0.020
ACCURACY_THRESHOLD = 0.60

#: Fourteen rules, none used in notes/16. Three demonstrations' worth per class
#: plus a held-out query member, matching the screen's fold shape.
RULES: dict[str, tuple[list[str], list[str]]] = {
    # Expected to clump.
    "colour_temp": (["scarlet", "amber", "crimson"], ["azure", "teal", "indigo"]),
    "body_vs_furniture": (["elbow", "kneecap", "shoulder"], ["dresser", "ottoman", "bookcase"]),
    "liquid_vs_solid": (["syrup", "vinegar", "kerosene"], ["granite", "plywood", "brick"]),
    "vehicle_vs_plant": (["tractor", "schooner", "glider"], ["fern", "clover", "bramble"]),
    "sentiment": (["radiant", "joyful", "splendid"], ["dreadful", "wretched", "ghastly"]),
    # Expected not to.
    "letter_count": (["table", "grape", "stone"], ["dolphin", "cabinet", "pelican"]),
    "ends_in_e": (["bridge", "candle", "marble"], ["basket", "lantern", "pillar"]),
    "double_letter": (["kettle", "rabbit", "puddle"], ["falcon", "ribbon", "marble"]),
    "multiple_of_three": (["9", "12", "21"], ["8", "11", "20"]),
    "prime": (["7", "13", "19"], ["8", "14", "20"]),
    # Honestly unsure -- these are what make it a test.
    "abstract_vs_concrete": (["justice", "freedom", "loyalty"], ["shovel", "saucer", "gravel"]),
    "singular_vs_plural": (["basket", "candle", "pillar"], ["baskets", "candles", "pillars"]),
    "past_vs_present": (["walked", "carried", "painted"], ["walks", "carries", "paints"]),
    "latin_vs_germanic": (["conclude", "provide", "reduce"], ["hearth", "thatch", "yonder"]),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fold(a: list[str], b: list[str]) -> tuple[list[str], list[str], str, str]:
    """Two demonstrations per class; the query is the third, never demonstrated."""
    return (a[:2], b[:2], a[2], b[2])


def measure(args: argparse.Namespace) -> None:
    if PROTOCOL.exists():
        raise SystemExit(f"protocol already frozen: {PROTOCOL}. Delete it only deliberately.")
    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

        predictions = {}
        for rule, (class_a, class_b) in RULES.items():
            per_layer = {}
            for layer in LAYERS:
                a = [embed(model, w, layer) for w in class_a]
                b = [embed(model, w, layer) for w in class_b]
                per_layer[f"layer_{layer}"] = separation(a, b)
            mean_sep = sum(per_layer.values()) / len(per_layer)
            predictions[rule] = {
                **per_layer,
                "mean_separation": mean_sep,
                "predicted_learnable": mean_sep >= SEPARATION_THRESHOLD,
            }
            print(f"{rule:22} sep {mean_sep:+.4f}  -> "
                  f"{'LEARNABLE' if mean_sep >= SEPARATION_THRESHOLD else 'not learnable'}",
                  flush=True)

        protocol = {
            "what_this_is": (
                "Predictions of rule learnability from class separation alone, frozen "
                "before any accuracy was measured. notes/19."
            ),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layers": list(LAYERS),
            "separation_threshold": SEPARATION_THRESHOLD,
            "accuracy_threshold": ACCURACY_THRESHOLD,
            "thresholds_derived_from": "notes/16 gaps: separation 0.008-0.043, accuracy 0.490-0.729",
            "n_rules": len(RULES),
            "n_predicted_learnable": sum(
                bool(v["predicted_learnable"]) for v in predictions.values()
            ),
            "predictions": predictions,
            "source_sha256": {
                "run_clustering_prediction.py": _sha256(Path(__file__)),
            },
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
        PROTOCOL.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
        print(f"\nfroze {PROTOCOL} — "
              f"{protocol['n_predicted_learnable']}/{len(RULES)} predicted learnable", flush=True)
    finally:
        model.free()


def test(args: argparse.Namespace) -> None:
    if not PROTOCOL.exists():
        raise SystemExit(
            f"no frozen protocol at {PROTOCOL}. Run --phase measure first; "
            "this phase must not be run before predictions exist."
        )
    protocol = json.loads(PROTOCOL.read_text())
    summary_path = OUT.with_name(OUT.stem.removesuffix("_raw") + "_summary.json")
    for path in (OUT, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    started = time.time()
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)
        design = cells()[:2] if args.smoke else cells()
        rows: list[dict[str, object]] = []

        for rule, (class_a, class_b) in RULES.items():
            arithmetic = rule in {"multiple_of_three", "prime"}
            a, b, query_a, query_b = _fold(class_a, class_b)
            for order, map_id, query_class in design:
                label_a, label_b = LABELS[map_id], LABELS[1 - map_id]
                pool, used = {0: list(a), 1: list(b)}, {0: 0, 1: 0}
                demos = []
                for which in order:
                    demos.append((pool[which][used[which]], label_a if which == 0 else label_b))
                    used[which] += 1
                query = query_a if query_class == 0 else query_b
                correct = label_a if query_class == 0 else label_b
                prompt = model.chat(render(demos, query, arithmetic), ANSWER_PREFIX)
                rows.append(
                    {
                        "rule": rule,
                        "fold": 0,
                        "order": list(order),
                        "map": map_id,
                        "query_class": query_class,
                        **score(model, prompt, correct),
                    }
                )
            print(f"{rule}: done", flush=True)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        results, hits = {}, 0
        for rule in RULES:
            sub = [r for r in rows if r["rule"] == rule]
            acc = sum(bool(r["correct"]) for r in sub) / len(sub)
            predicted = bool(protocol["predictions"][rule]["predicted_learnable"])
            observed = acc >= ACCURACY_THRESHOLD
            hit = predicted == observed
            hits += hit
            results[rule] = {
                "mean_separation": protocol["predictions"][rule]["mean_separation"],
                "predicted_learnable": predicted,
                "accuracy": acc,
                "twin_pair": twin_pair(sub),
                "observed_learnable": observed,
                "prediction_correct": hit,
            }

        summary = {
            "what_this_is": (
                "Frozen predictions from class separation, scored against measured "
                "accuracy. notes/19."
            ),
            "protocol": str(PROTOCOL),
            "protocol_sha256": _sha256(PROTOCOL),
            "model": MODEL,
            "smoke": bool(args.smoke),
            "separation_threshold": SEPARATION_THRESHOLD,
            "accuracy_threshold": ACCURACY_THRESHOLD,
            "n_rules": len(RULES),
            "n_correct_predictions": hits,
            "chance_correct": len(RULES) / 2,
            "by_rule": results,
            "misses": [r for r, v in results.items() if not v["prediction_correct"]],
            "elapsed_seconds": round(time.time() - started, 1),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"\n{hits}/{len(RULES)} predictions correct (chance {len(RULES)/2})", flush=True)
    finally:
        model.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("measure", "test"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    (measure if args.phase == "measure" else test)(args)


if __name__ == "__main__":
    main()
