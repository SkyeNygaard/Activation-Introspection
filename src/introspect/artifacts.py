"""Validation for matched-site profile bundles consumed by IFT runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from introspect.concepts import NEUTRAL_FILLERS, TEMPLATES
from introspect.ift import EVAL_VARIANTS
from introspect.probe import NATURAL_TEMPLATES
from introspect.prompts import IDENTIFY_FORCED_CHOICE_VARIANTS

MATCHED_ESTIMAND = "inject_at_layer_read_at_output"


def _artifact_path(summary_path: Path, reference: str) -> Path:
    raw_path = Path(reference)
    candidates = (
        [raw_path] if raw_path.is_absolute() else [summary_path.parent / raw_path, raw_path]
    )
    candidates = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"{summary_path} references missing artifact {reference!r}; "
        f"checked {[str(candidate) for candidate in candidates]}"
    )


def _verified_artifact(
    summary_path: Path, data: dict[str, Any], reference_key: str, checksum_key: str
) -> tuple[Path, bytes]:
    reference = data.get(reference_key)
    expected_sha = data.get(checksum_key)
    if not isinstance(reference, str) or not isinstance(expected_sha, str):
        raise ValueError(f"{summary_path} lacks {reference_key}/{checksum_key} provenance")
    artifact_path = _artifact_path(summary_path, reference)
    artifact_bytes = artifact_path.read_bytes()
    actual_sha = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"{artifact_path} checksum mismatch: {actual_sha} != {expected_sha}")
    return artifact_path, artifact_bytes


def _jsonl_records(path: Path, content: bytes) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(record)
    return records


def _mean_booleans(records: list[dict[str, Any]], key: str) -> float:
    values = [record.get(key) for record in records]
    if not values or any(not isinstance(value, bool) for value in values):
        raise ValueError(f"raw records lack boolean {key!r}")
    return sum(bool(value) for value in values) / len(values)


def _validate_natural_artifact(
    summary_path: Path, data: dict[str, Any], concepts: list[str]
) -> None:
    natural_path, natural_bytes = _verified_artifact(
        summary_path, data, "natural_trials", "natural_trials_sha256"
    )
    natural_records = _jsonl_records(natural_path, natural_bytes)
    readout = data.get("readout")
    if not isinstance(readout, dict):
        raise ValueError(f"{summary_path} lacks readout diagnostics")
    if len(natural_records) != int(readout.get("natural_train_n", -1)):
        raise ValueError(f"{summary_path} natural trial count mismatch")

    seen: set[tuple[str, int]] = set()
    for record in natural_records:
        try:
            identity = (str(record["concept"]), int(record["template_id"]))
            concept_class = int(record["concept_class"])
            prediction = int(record["grouped_cv_prediction_class"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{natural_path} has a malformed natural trial") from exc
        if identity in seen:
            raise ValueError(f"{natural_path} has duplicate natural trial {identity}")
        seen.add(identity)
        if identity[0] not in concepts or concept_class != concepts.index(identity[0]):
            raise ValueError(f"{natural_path} has inconsistent natural concept metadata")
        if not 0 <= identity[1] < len(NATURAL_TEMPLATES):
            raise ValueError(f"{natural_path} has an invalid natural template id")
        expected_template = NATURAL_TEMPLATES[identity[1]]
        if record.get("template") != expected_template or record.get(
            "rendered_text"
        ) != expected_template.format(concept=identity[0]):
            raise ValueError(f"{natural_path} has inconsistent natural-text provenance")
        if not 0 <= prediction < len(concepts):
            raise ValueError(f"{natural_path} has invalid grouped-CV prediction")
        correct = prediction == concept_class
        if record.get("grouped_cv_correct") is not correct:
            raise ValueError(f"{natural_path} has inconsistent grouped-CV correctness")

    natural_accuracy = _mean_booleans(natural_records, "grouped_cv_correct")
    if not math.isclose(
        float(readout.get("natural_grouped_cv_accuracy", float("nan"))),
        natural_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"{summary_path} natural grouped-CV accuracy mismatch")


def _validate_v2_profile(
    summary_path: Path,
    data: dict[str, Any],
    raw_path: Path,
    records: list[dict[str, Any]],
    concepts: list[str],
) -> None:
    """Cross-check v2 summaries against prediction-level artifacts."""
    null_permutations = data.get("null_permutations")
    if not isinstance(null_permutations, int) or null_permutations < 1:
        raise ValueError(f"{summary_path} has invalid null_permutations")

    prompt_provenance = data.get("prompt_provenance")
    if not isinstance(prompt_provenance, dict):
        raise ValueError(f"{summary_path} lacks prompt_provenance")
    expected_templates = {
        str(index): IDENTIFY_FORCED_CHOICE_VARIANTS[index] for index in EVAL_VARIANTS
    }
    if prompt_provenance.get("identify_variant_templates") != expected_templates:
        raise ValueError(f"{summary_path} uses incompatible evaluation prompt templates")
    expected_prompt_inputs = {
        "natural_templates": NATURAL_TEMPLATES,
        "concept_vector_templates": TEMPLATES,
        "concept_vector_neutral_fillers": NEUTRAL_FILLERS,
    }
    mismatched_prompt_inputs = [
        key
        for key, expected in expected_prompt_inputs.items()
        if prompt_provenance.get(key) != expected
    ]
    if mismatched_prompt_inputs:
        raise ValueError(
            f"{summary_path} uses incompatible prompt inputs: {', '.join(mismatched_prompt_inputs)}"
        )

    provenance = data.get("provenance")
    source_hashes = provenance.get("source_sha256") if isinstance(provenance, dict) else None
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(f"{summary_path} lacks source-code hashes")
    if any(not isinstance(value, str) or len(value) != 64 for value in source_hashes.values()):
        raise ValueError(f"{summary_path} has malformed source-code hashes")

    layers = data.get("layers")
    if not isinstance(layers, list):
        raise ValueError(f"{summary_path} lacks layer summaries")
    valid_rows = {
        int(row["injection_layer"]): row
        for row in layers
        if isinstance(row, dict)
        and row.get("valid", True)
        and row.get("probe_accuracy") is not None
    }
    records_by_layer: dict[int, list[dict[str, Any]]] = {}
    seen_trials: set[tuple[int, str, int]] = set()
    for record in records:
        try:
            layer = int(record["injection_layer"])
            concept = str(record["concept"])
            concept_class = int(record["concept_class"])
            trial_id = int(record["trial_id"])
            prediction = int(record["probe_prediction_class"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{raw_path} has a malformed injected-trial record") from exc
        identity = (layer, concept, trial_id)
        if identity in seen_trials:
            raise ValueError(f"{raw_path} has duplicate trial {identity}")
        seen_trials.add(identity)
        if concept not in concepts or concept_class != concepts.index(concept):
            raise ValueError(f"{raw_path} has inconsistent concept label for {concept!r}")
        if record.get("probe_correct") is not (prediction == concept_class):
            raise ValueError(f"{raw_path} has inconsistent probe correctness")
        option_order = record.get("option_order")
        try:
            target_index = int(record["target_option_index"])
            self_index = int(record["self_report_option_index"])
            prompt_variant = int(record["prompt_variant"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{raw_path} has malformed report metadata") from exc
        if (
            not isinstance(option_order, list)
            or any(not isinstance(option, str) for option in option_order)
            or sorted(option_order) != concepts
        ):
            raise ValueError(f"{raw_path} has an invalid option order")
        if not 0 <= target_index < len(concepts) or option_order[target_index] != concept:
            raise ValueError(f"{raw_path} has an inconsistent target option")
        if not 0 <= self_index < len(concepts):
            raise ValueError(f"{raw_path} has an invalid self-report option")
        if record.get("self_report_correct") is not (self_index == target_index):
            raise ValueError(f"{raw_path} has inconsistent self-report correctness")
        if prompt_variant not in EVAL_VARIANTS:
            raise ValueError(f"{raw_path} has an unexpected prompt variant")
        rendered_prompt = record.get("rendered_prompt")
        rendered_sha = record.get("rendered_prompt_sha256")
        if (
            not isinstance(rendered_prompt, str)
            or rendered_sha != hashlib.sha256(rendered_prompt.encode()).hexdigest()
        ):
            raise ValueError(f"{raw_path} has inconsistent rendered-prompt provenance")
        vector_sha = record.get("vector_sha256")
        if not isinstance(vector_sha, str) or len(vector_sha) != 64:
            raise ValueError(f"{raw_path} has malformed vector provenance")
        logprobs = record.get("self_report_digit_logprobs")
        probabilities = record.get("probe_class_probabilities")
        if not isinstance(logprobs, list) or len(logprobs) != len(concepts):
            raise ValueError(f"{raw_path} has malformed self-report logits")
        if not isinstance(probabilities, list) or len(probabilities) != len(concepts):
            raise ValueError(f"{raw_path} has malformed probe probabilities")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) for value in logprobs
        ):
            raise ValueError(f"{raw_path} has non-finite self-report logits")
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in probabilities
        ) or not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{raw_path} has invalid probe probabilities")
        null_predictions = record.get("null_probe_prediction_classes")
        null_correct = record.get("null_probe_correct")
        if not isinstance(null_predictions, list) or not isinstance(null_correct, list):
            raise ValueError(f"{raw_path} lacks null prediction records")
        if len(null_predictions) != null_permutations or len(null_correct) != null_permutations:
            raise ValueError(f"{raw_path} has the wrong number of null predictions")
        for null_prediction, correct in zip(null_predictions, null_correct, strict=True):
            if not isinstance(null_prediction, int) or not 0 <= null_prediction < len(concepts):
                raise ValueError(f"{raw_path} has an invalid null prediction")
            if not isinstance(correct, bool) or correct is not (null_prediction == concept_class):
                raise ValueError(f"{raw_path} has inconsistent null correctness")
        records_by_layer.setdefault(layer, []).append(record)

    if set(records_by_layer) != set(valid_rows):
        raise ValueError(
            f"{summary_path} raw/summary valid-layer mismatch: "
            f"{sorted(records_by_layer)} != {sorted(valid_rows)}"
        )
    for layer, row in valid_rows.items():
        layer_records = records_by_layer[layer]
        if len(layer_records) != int(row["n_trials"]):
            raise ValueError(f"{summary_path} raw trial count mismatch at L{layer}")
        probe_accuracy = _mean_booleans(layer_records, "probe_correct")
        self_accuracy = _mean_booleans(layer_records, "self_report_correct")
        null_by_permutation = [
            sum(bool(record["null_probe_correct"][idx]) for record in layer_records)
            / len(layer_records)
            for idx in range(null_permutations)
        ]
        pooled_null = sum(null_by_permutation) / null_permutations
        expected = {
            "probe_accuracy": probe_accuracy,
            "self_report_accuracy": self_accuracy,
            "permuted_label_accuracy": pooled_null,
        }
        for key, value in expected.items():
            if not math.isclose(float(row[key]), value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{summary_path} {key} disagrees with raw trials at L{layer}")
        saved_null = row.get("permuted_label_accuracy_by_permutation")
        if not isinstance(saved_null, list) or len(saved_null) != null_permutations:
            raise ValueError(f"{summary_path} lacks per-permutation null accuracy at L{layer}")
        if any(
            not math.isclose(float(saved), observed, rel_tol=0.0, abs_tol=1e-12)
            for saved, observed in zip(saved_null, null_by_permutation, strict=True)
        ):
            raise ValueError(f"{summary_path} null summaries disagree at L{layer}")
        bank_hashes = row.get("bank_vector_sha256")
        if not isinstance(bank_hashes, dict):
            raise ValueError(f"{summary_path} lacks bank vector hashes at L{layer}")
        for record in layer_records:
            if bank_hashes.get(record["concept"]) != record["vector_sha256"]:
                raise ValueError(f"{summary_path} vector hash disagrees at L{layer}")

    _validate_natural_artifact(summary_path, data, concepts)


def load_matched_profile(
    path: Path,
    *,
    expected_model: str,
    expected_strength: float,
    expected_concepts: list[str],
    expected_n_layers: int,
    expected_model_revision: str | None,
    allow_legacy_schema1: bool = False,
    require_model_revision: bool = True,
) -> dict[int, float]:
    """Load a compatible matched profile and verify its raw artifacts.

    Schema 1 can be inspected only through an explicit opt-in. It lacks the
    natural-readout, prediction-level, and source-hash checks required for an IFT
    comparison, so experiment runners must keep the default strict behavior.
    """
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} is not a JSON object")
    data: dict[str, Any] = loaded
    schema_version = data.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError(f"{path} has unsupported schema_version {schema_version!r}")
    if schema_version == 1 and not allow_legacy_schema1:
        raise ValueError(
            f"{path} uses legacy schema_version 1, which is insufficient for a new "
            "IFT comparison; regenerate it with scripts/run_reach_output.py"
        )
    if data.get("estimand") != MATCHED_ESTIMAND:
        raise ValueError(
            f"{path} is not a matched-site profile: expected estimand "
            f"{MATCHED_ESTIMAND!r}, got {data.get('estimand')!r}"
        )
    stored_strength = data.get("strength")
    stored_concepts = data.get("concepts")
    checks = {
        "model": data.get("model") == expected_model,
        "strength": (
            isinstance(stored_strength, (int, float))
            and math.isclose(float(stored_strength), expected_strength, rel_tol=0.0, abs_tol=1e-12)
        ),
        "concepts": (
            isinstance(stored_concepts, list)
            and all(isinstance(concept, str) for concept in stored_concepts)
            and sorted(stored_concepts) == sorted(expected_concepts)
        ),
        "n_model_layers": data.get("n_model_layers") == expected_n_layers,
        "output_layer": data.get("output_layer") == expected_n_layers - 1,
    }
    mismatched = [name for name, matched in checks.items() if not matched]
    if mismatched:
        raise ValueError(f"{path} is incompatible on: {', '.join(mismatched)}")
    recorded_revision = data.get("model_revision")
    if require_model_revision and (not expected_model_revision or not recorded_revision):
        raise ValueError(f"{path} cannot be joined without immutable model revisions on both runs")
    if expected_model_revision and recorded_revision != expected_model_revision:
        raise ValueError(
            f"{path} model revision mismatch: {recorded_revision!r} != {expected_model_revision!r}"
        )

    raw_path, raw_bytes = _verified_artifact(path, data, "raw_trials", "raw_trials_sha256")
    raw_records = _jsonl_records(raw_path, raw_bytes)
    layers = data.get("layers")
    if not isinstance(layers, list) or any(not isinstance(row, dict) for row in layers):
        raise ValueError(f"{path} lacks valid layer summaries")
    layer_rows = [row for row in layers if isinstance(row, dict)]
    profile = {
        int(row["injection_layer"]): float(row["probe_accuracy"])
        for row in layer_rows
        if row.get("valid", True) and row.get("probe_accuracy") is not None
    }
    if not profile:
        raise ValueError(f"{path} has no valid matched-site layer estimates")

    if schema_version == 2:
        _validate_v2_profile(path, data, raw_path, raw_records, sorted(expected_concepts))
    else:
        raw_counts: dict[int, int] = {}
        for record in raw_records:
            try:
                layer = int(record["injection_layer"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{raw_path} has a malformed trial record") from exc
            raw_counts[layer] = raw_counts.get(layer, 0) + 1
        for row in layer_rows:
            if row.get("valid", True) and row.get("probe_accuracy") is not None:
                layer = int(row["injection_layer"])
                if raw_counts.get(layer, 0) != int(row["n_trials"]):
                    raise ValueError(
                        f"{path} raw trial count mismatch at L{layer}: "
                        f"{raw_counts.get(layer, 0)} != {row['n_trials']}"
                    )
    return profile
