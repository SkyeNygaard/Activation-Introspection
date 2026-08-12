"""Does introspection training beat reading the same states with a probe?

Belinda Li's project description says supervision for verbalization training
"comes cheaply from the internals themselves: probe readouts, feature activation
values, or the measured effects of ablation and patching." That makes one
comparison unavoidable: if a probe fit on the same states the adapter trained on
does as well as the adapter, then introspection training is probe distillation
with extra steps.

[`notes/07`](../notes/07-trained-activation-reporter.md) reports the adapter at
0.927 mean twin-pair accuracy over four seeds, range [0.833, 1.000], on eight
concept directions and three carriers withheld from training. This fits readers
on the adapter's own training bank and evaluates them on the adapter's own
evaluation bank.

Unlike [`notes/11`](../notes/11-matched-cost-reader.md), the outcome here is not
obvious. There the reader saw four labelled demonstrations of the *same*
direction it was asked about. Here it must generalize the sign-reading operation
to directions it has never seen, and the concept bank is explicitly built to be
far from collinear. A linear reader may simply not transfer.

Three readers, in increasing order of unfair advantage:

* ``centroid`` -- difference of means over the training states. The canonical
  cheap probe, and what "probe readouts" means in the project description.
* ``logistic`` -- L2-regularized logistic regression on the same states. The
  strongest simple reader, included so a negative cannot be blamed on using a
  weak probe.
* ``oracle_direction`` -- told the held-out concept direction and asked only for
  the sign of the projection. Not a fair third party; an upper bound that
  brackets where the adapter sits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect import models
from introspect.concepts import ConceptVector, build_bank
from introspect.hooks import capture, intervene
from introspect.preflight import check as preflight_check
from introspect.report_training import (
    CENTERING_CONCEPTS,
    EVAL_CARRIERS,
    EVAL_CONCEPTS,
    TRAIN_CARRIERS,
    TRAIN_CONCEPTS,
    prepare_report,
    sha256_text,
    sign_intervention,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LAYER = 9
STRENGTH = 1.0
#: notes/07, four seeds, twin-pair accuracy on the same evaluation bank.
ADAPTER_MEAN = 0.927
ADAPTER_RANGE = (0.833, 1.000)
READERS = ("centroid", "logistic", "oracle_direction", "shuffled_labels")
SOURCE_PATHS = (
    "scripts/run_trained_vs_probe.py",
    "src/introspect/report_training.py",
    "src/introspect/concepts.py",
    "src/introspect/hooks.py",
    "src/introspect/models.py",
    "pyproject.toml",
    "uv.lock",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _protocol() -> dict[str, object]:
    return {
        "schema_version": 1,
        "frozen_on": "2026-08-11",
        "question": (
            "Does the trained activation reporter exceed a probe fitted on the same "
            "states it trained on, when both are asked about held-out concept "
            "directions?"
        ),
        "why_this_is_not_a_repeat_of_notes_11": (
            "In notes/11 the reader saw four labelled demonstrations of the same "
            "direction it was queried on, and won at 1.000. Here every reader must "
            "transfer to eight directions it has never seen, from a bank built to be "
            "far from collinear. A linear reader has no guarantee of transferring, so "
            "the direction of this result is genuinely open."
        ),
        "design": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "strength": STRENGTH,
            "train_concepts": list(TRAIN_CONCEPTS),
            "train_carriers": list(TRAIN_CARRIERS),
            "eval_concepts": list(EVAL_CONCEPTS),
            "eval_carriers": list(EVAL_CARRIERS),
            "centering_concepts": list(CENTERING_CONCEPTS),
            "readers": list(READERS),
            "reader_training_data": (
                "the residual at the marker position under the same sign intervention "
                "the adapter trained on: 8 train concepts x 6 train carriers x 2 signs"
            ),
            "evaluation": (
                "8 eval concepts x 3 eval carriers x 2 signs = 24 twin pairs, the "
                "adapter's own evaluation bank"
            ),
            "unit": "twin pair; a cell counts only if both signs are read correctly",
        },
        "comparator": {
            "adapter_mean_twin_pair": ADAPTER_MEAN,
            "adapter_seed_range": list(ADAPTER_RANGE),
            "note": (
                "four seeds give a mean and a range, not an interval; a reader inside "
                "the range is not distinguishable from the adapter by this evidence"
            ),
        },
        "prereg_fork": {
            "probe_at_or_above_adapter_mean": (
                "introspection training does not exceed reading the same states. The "
                "0.927 is then probe-equivalent at best, and the training result must "
                "be described as distillation of an available linear signal rather "
                "than as taught introspection."
            ),
            "probe_below_adapter_range": (
                "the trained model generalizes the sign-reading operation to unseen "
                "directions in a way a matched linear reader does not. That would be "
                "the first result in this repository to survive the equal-or-lower-cost "
                "criterion, and it must then be checked against the oracle upper bound "
                "before any privileged-access language is used."
            ),
            "probe_inside_adapter_range": (
                "report as indistinguishable. Do not call it a win for either side."
            ),
        },
        "sanity_gate": {
            "shuffled_labels_row_accuracy_band": [0.35, 0.65],
            "rule": (
                "a probe fitted on permuted training signs must not read the held-out "
                "bank. The gate is on ROW accuracy, over 48 rows, where the coin-flip "
                "null is 0.500 with SD 0.072, so the band is about +/- 2 SD."
            ),
            "disclosed_change": (
                "v1 gated on twin-pair accuracy at 0.30. That is miscalibrated: the "
                "twin-pair null is 0.25 with SD 0.088 over 24 pairs, so a threshold of "
                "0.30 sits 0.57 SD above the null and fails about a quarter of the time "
                "on a perfectly good control. The smoke tripped it at 3/6 pairs, which "
                "prompted re-deriving the null rather than accepting the verdict. The "
                "statistic moved to the better-powered one before any confirmatory run; "
                "twin-pair for the control is still reported, descriptively."
            ),
        },
        "oracle_role": (
            "upper bound only, never a gate. It is told the held-out direction, which "
            "no fair third party has, and exists to bracket the adapter."
        ),
        "source_files_sha256": {path: _sha256(ROOT / path) for path in SOURCE_PATHS},
    }


def _freeze_protocol(path: Path) -> tuple[dict[str, object], str]:
    protocol = _protocol()
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise SystemExit(f"{path} differs from this source; issue a new protocol version")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol, _sha256(path)


@torch.no_grad()
def _state(
    model: models.LoadedModel, carrier: str, direction: ConceptVector | None, sign: int
) -> Tensor:
    """The residual at the marker position under one signed intervention."""
    prepared = prepare_report(model, carrier)
    interventions = (
        []
        if direction is None
        else sign_intervention(
            direction.vector,
            LAYER,
            prepared.marker_position,
            sign,
            strength=STRENGTH,
            label=f"probe:{direction.name}:{sign:+d}",
        )
    )
    with (
        intervene(model, interventions, prompt_len=int(prepared.input_ids.shape[1])),
        capture(model, [LAYER]) as store,
    ):
        model.forward_logits(prepared.input_ids)
    return store.acts[LAYER][0][0, prepared.marker_position].clone()


def _fit_logistic(features: Tensor, labels: Tensor, steps: int = 400) -> Tensor:
    """L2-regularized logistic regression, full batch. Returns the weight vector."""
    weight = torch.zeros(features.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.05)
    target = (labels > 0).float()
    for _ in range(steps):
        optimizer.zero_grad()
        logits = features @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss = loss + 1e-3 * weight.pow(2).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    return torch.cat([weight.detach(), bias.detach()])


def _twin_pair(correct: dict[tuple[str, str, int], bool]) -> float:
    cells = {(concept, carrier) for concept, carrier, _ in correct}
    return sum(
        correct[(concept, carrier, 1)] and correct[(concept, carrier, -1)]
        for concept, carrier in cells
    ) / len(cells)


def run(args: argparse.Namespace) -> None:
    out = args.out
    manifest_path = out.with_suffix(".manifest.json")
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    for path in (out, manifest_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    protocol, protocol_sha = _freeze_protocol(
        args.protocol or Path("results/trained_vs_probe_protocol_v1.json")
    )
    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    started = time.time()
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

        train_concepts = TRAIN_CONCEPTS[:2] if args.smoke else TRAIN_CONCEPTS
        eval_concepts = EVAL_CONCEPTS[:2] if args.smoke else EVAL_CONCEPTS
        train_bank = centered(train_concepts)
        eval_bank = centered(eval_concepts)

        train_states, train_signs = [], []
        for concept in train_concepts:
            for carrier in TRAIN_CARRIERS:
                for sign in (1, -1):
                    train_states.append(_state(model, carrier, train_bank[concept], sign))
                    train_signs.append(sign)
            print(f"train states: {concept}", flush=True)
        features = torch.stack(train_states)
        signs = torch.tensor(train_signs, dtype=torch.float32)

        positive = features[signs > 0].mean(0)
        negative = features[signs < 0].mean(0)
        logistic = _fit_logistic(features, signs)
        permuted = signs[torch.randperm(len(signs), generator=torch.Generator().manual_seed(0))]
        shuffled = _fit_logistic(features, permuted)

        clean = {carrier: _state(model, carrier, None, 1) for carrier in EVAL_CARRIERS}

        rows: list[dict[str, object]] = []
        correct: dict[str, dict[tuple[str, str, int], bool]] = {r: {} for r in READERS}
        for concept in eval_concepts:
            for carrier in EVAL_CARRIERS:
                for sign in (1, -1):
                    state = _state(model, carrier, eval_bank[concept], sign)
                    unit = eval_bank[concept].vector / eval_bank[concept].vector.norm()
                    predictions = {
                        "centroid": 1
                        if (state - positive).norm() <= (state - negative).norm()
                        else -1,
                        "logistic": 1 if float(state @ logistic[:-1] + logistic[-1]) >= 0 else -1,
                        "oracle_direction": 1
                        if float((state - clean[carrier]) @ unit) >= 0
                        else -1,
                        "shuffled_labels": 1
                        if float(state @ shuffled[:-1] + shuffled[-1]) >= 0
                        else -1,
                    }
                    for reader, predicted in predictions.items():
                        correct[reader][(concept, carrier, sign)] = predicted == sign
                    rows.append(
                        {
                            "concept": concept,
                            "carrier_sha256": sha256_text(carrier),
                            "sign": sign,
                            "state_norm": float(state.norm()),
                            "predictions": predictions,
                            "correct": {r: predictions[r] == sign for r in READERS},
                        }
                    )
            print(f"eval states: {concept}", flush=True)

        row_accuracy = {
            reader: sum(correct[reader].values()) / len(correct[reader]) for reader in READERS
        }
        twin = {reader: _twin_pair(correct[reader]) for reader in READERS}
        best = max(twin["centroid"], twin["logistic"])
        summary = {
            "n_states": len(rows),
            "n_twin_pairs": len(rows) // 2,
            "row_accuracy": row_accuracy,
            "twin_pair_accuracy": twin,
            "best_fair_reader_twin_pair": best,
            "adapter_mean_twin_pair": ADAPTER_MEAN,
            "adapter_seed_range": list(ADAPTER_RANGE),
            "adapter_minus_best_fair_reader": ADAPTER_MEAN - best,
            "shuffled_labels_twin_pair_descriptive": twin["shuffled_labels"],
            "twin_pair_coin_flip_null": 0.25,
            "gates": {"shuffled_labels_at_chance": 0.35 <= row_accuracy["shuffled_labels"] <= 0.65},
            "verdict": (
                "void_sanity_gate_failed"
                if not 0.35 <= row_accuracy["shuffled_labels"] <= 0.65
                else "training_does_not_exceed_a_probe"
                if best >= ADAPTER_MEAN
                else "training_exceeds_the_matched_probe"
                if best < ADAPTER_RANGE[0]
                else "indistinguishable_within_seed_range"
            ),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp")
        with tmp.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        tmp.replace(out)
        raw_sha = _sha256(out)
        config = {
            "schema_version": 1,
            "model": model.name,
            "model_revision": models.loaded_revision(model),
            "device": str(model.device),
            "dtype": str(model.dtype),
            "layer": LAYER,
            "strength": STRENGTH,
            "protocol": protocol,
            "protocol_sha256": protocol_sha,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        }
        for path, value in (
            (
                manifest_path,
                {
                    "schema_version": 1,
                    "config": config,
                    "config_sha256": _json_sha256(config),
                    "raw": out.name,
                    "raw_sha256": raw_sha,
                    "n_rows": len(rows),
                    "elapsed_seconds": time.time() - started,
                },
            ),
            (summary_path, summary),
        ):
            tmp = path.with_name(f".{path.name}.tmp")
            tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            tmp.replace(path)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"wrote {out} ({raw_sha})", flush=True)
    finally:
        model.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/trained_vs_probe_v1_raw.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
