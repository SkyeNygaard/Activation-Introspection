"""Matched-site reach-to-output profile with raw trials and provenance.

For each layer L, construct concept vectors at L, inject at L, and read the
final residual with one natural-text-trained classifier. This matches the
injection site used by ``run_ift.py``. It is deliberately different from
``layer_profile.py``, which holds the injection source fixed and measures how a
single intervention propagates through depth.

The output is descriptive. Layers are ordered, dependent sites rather than
independent samples, and reaching the output residual is not by itself evidence
of causal use or introspection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from introspect.concepts import (
    DEFAULT_CONCEPTS,
    NEUTRAL_FILLERS,
    TEMPLATES,
    ConceptVector,
    build_bank,
    max_offdiagonal_cosine,
)
from introspect.grading import score_choices
from introspect.hooks import Intervention, capture, intervene
from introspect.ift import EVAL_VARIANTS, seeds_for
from introspect.models import DEFAULT_MODEL, LoadedModel, load
from introspect.probe import NATURAL_TEMPLATES, collect_natural
from introspect.prompts import (
    IDENTIFY_FORCED_CHOICE_VARIANTS,
    forced_choice,
    permuted_options,
    variant,
)

console = Console()
ESTIMAND = "inject_at_layer_read_at_output"
SCHEMA_VERSION = 2

CODE_PATHS = (
    "scripts/run_reach_output.py",
    "src/introspect/concepts.py",
    "src/introspect/grading.py",
    "src/introspect/hooks.py",
    "src/introspect/ift.py",
    "src/introspect/models.py",
    "src/introspect/probe.py",
    "src/introspect/prompts.py",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_sha256(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, default=str).encode())


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_hashes() -> dict[str, str]:
    """Hash executable inputs so a dirty-tree run remains auditable."""
    root = Path(__file__).resolve().parents[1]
    return {relative: _sha256((root / relative).read_bytes()) for relative in CODE_PATHS}


def _tensor_sha256(vector: torch.Tensor) -> str:
    array = vector.detach().to("cpu", torch.float32).contiguous().numpy()
    return _sha256(array.tobytes())


def _array_bundle_sha256(*arrays: np.ndarray) -> str:
    payload = b"".join(np.asarray(array).tobytes() for array in arrays)
    return _sha256(payload)


def _artifact_reference(artifact: Path, summary: Path) -> str:
    """Store a path relative to the summary for portable artifact bundles."""
    return os.path.relpath(artifact.resolve(), start=summary.resolve().parent)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def grouped_natural_predictions(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return held-template predictions and fold ids for audit diagnostics."""
    predictions = np.full(len(y), -1, dtype=int)
    fold_ids = np.full(len(y), -1, dtype=int)
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold_id, (train, test) in enumerate(cv.split(x, y, groups)):
        fold_scaler = StandardScaler().fit(x[train])
        fold_readout = LogisticRegression(max_iter=3000).fit(
            fold_scaler.transform(x[train]), y[train]
        )
        predictions[test] = cast(np.ndarray, fold_readout.predict(fold_scaler.transform(x[test])))
        fold_ids[test] = fold_id
    if (predictions < 0).any() or (fold_ids < 0).any():
        raise RuntimeError("grouped natural CV did not assign every observation")
    return predictions, fold_ids


@torch.no_grad()
def collect_layer_trials(
    model: LoadedModel,
    bank: dict[str, ConceptVector],
    injection_layer: int,
    output_layer: int,
    strength: float,
    trial_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    """Capture output activations and one raw record per injected trial."""
    vector_layers = {cv.layer for cv in bank.values()}
    if vector_layers != {injection_layer}:
        raise ValueError(
            f"injection at L{injection_layer} requires an L{injection_layer} bank; "
            f"got {sorted(vector_layers)}"
        )

    concepts = sorted(bank)
    digits = [str(i + 1) for i in range(len(concepts))]
    features: list[np.ndarray] = []
    labels: list[int] = []
    records: list[dict[str, object]] = []

    was_training = model.model.training
    model.model.eval()
    try:
        for concept_idx, name in enumerate(concepts):
            iv = Intervention(
                layer=injection_layer,
                direction=bank[name].vector,
                strength=strength,
            )
            for trial_id in trial_ids:
                options = permuted_options(concepts, trial_id)
                target_index = options.index(name)
                prompt_variant = trial_id % len(IDENTIFY_FORCED_CHOICE_VARIANTS)
                prompt = model.chat(
                    variant(IDENTIFY_FORCED_CHOICE_VARIANTS, trial_id).format(
                        options=forced_choice(options)
                    )
                )
                ids = model.encode(prompt)
                with (
                    intervene(model, [iv], prompt_len=int(ids.shape[1])),
                    capture(model, [output_layer]) as store,
                ):
                    choice = score_choices(model, prompt, digits)

                features.append(store.last_token(output_layer)[0].numpy())
                labels.append(concept_idx)
                records.append(
                    {
                        "injection_layer": injection_layer,
                        "output_layer": output_layer,
                        "concept": name,
                        "concept_class": concept_idx,
                        "trial_id": trial_id,
                        "prompt_variant": prompt_variant,
                        "option_order": options,
                        "target_option_index": target_index,
                        "self_report_option_index": choice.argmax,
                        "self_report_correct": choice.argmax == target_index,
                        "self_report_digit_logprobs": choice.logprobs,
                        "rendered_prompt": prompt,
                        "rendered_prompt_sha256": _sha256(prompt.encode()),
                        "vector_sha256": _tensor_sha256(bank[name].vector),
                    }
                )
    finally:
        model.model.train(was_training)

    return np.stack(features), np.array(labels), records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--strength", type=float, default=0.2)
    ap.add_argument(
        "--layers",
        default="",
        help="comma-separated injection layers; default is every layer except embedding",
    )
    ap.add_argument(
        "--orders-per-variant",
        type=int,
        default=10,
        help="option permutations for each held-out IFT prompt variant",
    )
    ap.add_argument("--null-permutations", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None, help="summary JSON path")
    ap.add_argument("--raw-out", type=Path, default=None, help="raw trial JSONL path")
    ap.add_argument(
        "--natural-raw-out", type=Path, default=None, help="natural-readout trial JSONL path"
    )
    args = ap.parse_args()
    if args.orders_per_variant < 1:
        ap.error("--orders-per-variant must be positive")
    if args.null_permutations < 1:
        ap.error("--null-permutations must be positive")

    model = load(args.model)
    output_layer = model.n_layers - 1
    layers = (
        [int(value) for value in args.layers.split(",") if value.strip()]
        if args.layers
        else list(range(1, model.n_layers))
    )
    if not layers or any(layer < 0 or layer >= model.n_layers for layer in layers):
        ap.error(f"layers must fall in [0, {model.n_layers - 1}]")
    if len(layers) != len(set(layers)):
        ap.error("--layers must not contain duplicates")

    concepts = sorted(DEFAULT_CONCEPTS)
    trial_ids = seeds_for(EVAL_VARIANTS, args.orders_per_variant)
    console.print(
        f"{model.name}; inject at each of {len(layers)} sites; read L{output_layer}; "
        f"{len(trial_ids)} option orders per concept"
    )

    # The readout is trained once on natural text at the one shared read site.
    x_nat, y_nat, natural_groups = collect_natural(model, concepts, output_layer)
    scaler = StandardScaler().fit(x_nat)
    xs_nat = scaler.transform(x_nat)
    readout = LogisticRegression(max_iter=3000).fit(xs_nat, y_nat)
    natural_predictions, natural_fold_ids = grouped_natural_predictions(
        x_nat, y_nat, natural_groups
    )
    natural_hits = natural_predictions == y_nat
    natural_train_predictions = cast(np.ndarray, readout.predict(xs_nat))
    natural_records: list[dict[str, object]] = []
    natural_index = 0
    for concept_class, concept in enumerate(concepts):
        for template_id, template in enumerate(NATURAL_TEMPLATES):
            natural_records.append(
                {
                    "concept": concept,
                    "concept_class": concept_class,
                    "template_id": template_id,
                    "template": template,
                    "rendered_text": template.format(concept=concept),
                    "grouped_cv_fold": int(natural_fold_ids[natural_index]),
                    "grouped_cv_prediction_class": int(natural_predictions[natural_index]),
                    "grouped_cv_prediction_concept": concepts[
                        int(natural_predictions[natural_index])
                    ],
                    "grouped_cv_correct": bool(natural_hits[natural_index]),
                    "in_sample_prediction_class": int(natural_train_predictions[natural_index]),
                    "in_sample_correct": bool(
                        natural_train_predictions[natural_index] == concept_class
                    ),
                }
            )
            natural_index += 1

    rng = np.random.default_rng(0)
    null_readouts: list[LogisticRegression] = []
    null_training_label_sha256: list[str] = []
    for _ in range(args.null_permutations):
        permuted_labels = rng.permutation(y_nat)
        null_training_label_sha256.append(_sha256(permuted_labels.astype(np.int64).tobytes()))
        null_readouts.append(LogisticRegression(max_iter=3000).fit(xs_nat, permuted_labels))

    raw_records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    table = Table("inject L", "bank cosine", "n", "probe", "permuted null", "self-report")
    for layer in layers:
        bank = build_bank(model, layer, concepts=concepts)
        bank_cosine = max_offdiagonal_cosine(bank)
        if bank_cosine >= 0.5:
            summaries.append(
                {
                    "injection_layer": layer,
                    "valid": False,
                    "reason": "degenerate_concept_bank",
                    "bank_max_abs_offdiag_cosine": bank_cosine,
                }
            )
            table.add_row(str(layer), f"{bank_cosine:.3f}", "-", "INVALID", "-", "-")
            continue

        x_inj, y_inj, records = collect_layer_trials(
            model, bank, layer, output_layer, args.strength, trial_ids
        )
        xs_inj = scaler.transform(x_inj)
        probe_predictions = cast(np.ndarray, readout.predict(xs_inj))
        probe_probabilities = cast(np.ndarray, readout.predict_proba(xs_inj))
        null_predictions = [cast(np.ndarray, clf.predict(xs_inj)) for clf in null_readouts]
        probe_hits = probe_predictions == y_inj
        null_correct = [prediction == y_inj for prediction in null_predictions]
        null_hits = np.concatenate(null_correct)
        self_hits = np.array([bool(record["self_report_correct"]) for record in records])

        for trial_index, (record, prediction, probabilities, correct) in enumerate(
            zip(records, probe_predictions, probe_probabilities, probe_hits, strict=True)
        ):
            record["probe_prediction_class"] = int(prediction)
            record["probe_prediction_concept"] = concepts[int(prediction)]
            record["probe_class_probabilities"] = [float(value) for value in probabilities]
            record["probe_correct"] = bool(correct)
            record["null_probe_prediction_classes"] = [
                int(values[trial_index]) for values in null_predictions
            ]
            record["null_probe_correct"] = [bool(values[trial_index]) for values in null_correct]
            raw_records.append(record)

        summary = {
            "injection_layer": layer,
            "valid": True,
            "bank_max_abs_offdiag_cosine": bank_cosine,
            "n_trials": len(records),
            "probe_accuracy": float(probe_hits.mean()),
            "permuted_label_accuracy": float(null_hits.mean()),
            "permuted_label_accuracy_by_permutation": [
                float(values.mean()) for values in null_correct
            ],
            "self_report_accuracy": float(self_hits.mean()),
            "bank_vector_sha256": {
                concept: _tensor_sha256(bank[concept].vector) for concept in concepts
            },
        }
        summaries.append(summary)
        table.add_row(
            str(layer),
            f"{bank_cosine:.3f}",
            str(len(records)),
            f"{summary['probe_accuracy']:.3f}",
            f"{summary['permuted_label_accuracy']:.3f}",
            f"{summary['self_report_accuracy']:.3f}",
        )

    console.print(table)

    safe_name = args.model.replace("/", "-")
    summary_path = args.out or Path("results") / f"reach_output_{safe_name}.summary.json"
    raw_dir = summary_path.parent / "raw"
    artifact_stem = summary_path.name.removesuffix(".summary.json")
    raw_path = args.raw_out or raw_dir / f"{artifact_stem}.trials.jsonl"
    natural_raw_path = args.natural_raw_out or raw_dir / f"{artifact_stem}.natural-trials.jsonl"
    if len({summary_path.resolve(), raw_path.resolve(), natural_raw_path.resolve()}) != 3:
        ap.error("--out, --raw-out, and --natural-raw-out must be distinct paths")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    natural_raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in raw_records)
    raw_path.write_text(raw_text)
    natural_raw_text = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in natural_records
    )
    natural_raw_path.write_text(natural_raw_text)

    config = cast(Any, model.model).config
    config_dict = config.to_dict() if hasattr(config, "to_dict") else vars(config)
    tokenizer = cast(Any, model.tokenizer)
    chat_template = getattr(tokenizer, "chat_template", None)
    per_concept_accuracy = {
        concept: float(natural_hits[y_nat == concept_class].mean())
        for concept_class, concept in enumerate(concepts)
    }
    per_template_accuracy = {
        str(template_id): float(natural_hits[natural_groups == template_id].mean())
        for template_id in sorted(set(int(value) for value in natural_groups))
    }
    confusion = [
        [
            int(np.sum((y_nat == actual) & (natural_predictions == predicted)))
            for predicted in range(len(concepts))
        ]
        for actual in range(len(concepts))
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "estimand": ESTIMAND,
        "status": "descriptive_not_confirmatory",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model": model.name,
        "model_revision": getattr(config, "_commit_hash", None),
        "n_model_layers": model.n_layers,
        "output_layer": output_layer,
        "strength": args.strength,
        "concepts": concepts,
        "prompt_variants": list(EVAL_VARIANTS),
        "option_orders_per_variant": args.orders_per_variant,
        "trial_ids": trial_ids,
        "null_permutations": args.null_permutations,
        "null_training_label_seed": 0,
        "null_training_label_sha256": null_training_label_sha256,
        "readout": {
            "type": "StandardScaler + LogisticRegression",
            "logistic_regression": {"max_iter": 3000, "library_defaults_otherwise": True},
            "natural_train_n": len(y_nat),
            "natural_templates_n": len(NATURAL_TEMPLATES),
            "natural_grouped_cv_folds": min(5, len(set(int(v) for v in natural_groups))),
            "natural_grouping_unit": "template_id",
            "natural_grouped_cv_accuracy": float(natural_hits.mean()),
            "natural_in_sample_accuracy": float((natural_train_predictions == y_nat).mean()),
            "natural_grouped_cv_accuracy_by_concept": per_concept_accuracy,
            "natural_grouped_cv_accuracy_by_template": per_template_accuracy,
            "natural_grouped_cv_confusion_matrix": confusion,
            "scaler_and_readout_parameter_sha256": _array_bundle_sha256(
                scaler.mean_,
                scaler.scale_,
                readout.classes_,
                readout.coef_,
                readout.intercept_,
            ),
            "null_readout_parameter_sha256": [
                _array_bundle_sha256(clf.classes_, clf.coef_, clf.intercept_)
                for clf in null_readouts
            ],
        },
        "layers": summaries,
        "raw_trials": _artifact_reference(raw_path, summary_path),
        "raw_trials_sha256": _sha256(raw_text.encode()),
        "natural_trials": _artifact_reference(natural_raw_path, summary_path),
        "natural_trials_sha256": _sha256(natural_raw_text.encode()),
        "prompt_provenance": {
            "identify_variant_indices": list(EVAL_VARIANTS),
            "identify_variant_templates": {
                str(index): IDENTIFY_FORCED_CHOICE_VARIANTS[index] for index in EVAL_VARIANTS
            },
            "natural_templates": NATURAL_TEMPLATES,
            "concept_vector_templates": TEMPLATES,
            "concept_vector_neutral_fillers": NEUTRAL_FILLERS,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
            "chat_template": chat_template,
            "chat_template_sha256": (
                _sha256(chat_template.encode()) if isinstance(chat_template, str) else None
            ),
        },
        "provenance": {
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "source_sha256": _source_hashes(),
            "argv": sys.argv,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _version("transformers"),
            "scikit_learn": _version("scikit-learn"),
            "device": str(model.device),
            "dtype": str(model.dtype),
            "model_config_sha256": _json_sha256(config_dict),
            "tokenizer_init_kwargs_sha256": _json_sha256(getattr(tokenizer, "init_kwargs", {})),
            "randomness": {
                "torch_manual_seed": 0,
                "null_label_numpy_seed": 0,
                "option_order": "random.Random(trial_id)",
                "torch_deterministic_algorithms_enabled": (
                    torch.are_deterministic_algorithms_enabled()
                ),
            },
        },
        "limitations": [
            "Layers are dependent ordered sites, not independent correlation samples.",
            "Repeated option orders are nuisance marginalization, not model seeds.",
            "Output decodability does not establish causal use or introspective access.",
            "If model_revision is missing, the model-config hash does not identify "
            "immutable weights.",
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    console.print(f"wrote {summary_path}, {raw_path}, and {natural_raw_path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
