"""Regression tests for IFT validity failures found during the claim audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

import introspect.ift as ift
from introspect.artifacts import MATCHED_ESTIMAND, load_matched_profile
from introspect.concepts import NEUTRAL_FILLERS, TEMPLATES, ConceptVector
from introspect.figures import plot_ift_vs_probe
from introspect.models import LoadedModel
from introspect.probe import NATURAL_TEMPLATES
from introspect.prompts import IDENTIFY_FORCED_CHOICE_VARIANTS, permuted_options


class _ChatOnly:
    def chat(self, user: str) -> str:
        return user


def _bank(layer: int, names: tuple[str, ...] = ("bread", "ocean")) -> dict[str, ConceptVector]:
    return {
        name: ConceptVector(name=name, layer=layer, vector=torch.eye(len(names))[idx])
        for idx, name in enumerate(names)
    }


def test_option_permutations_are_reproducible_and_move_every_target() -> None:
    concepts = ["bread", "clock", "ocean", "spider"]
    assert permuted_options(concepts, 7) == permuted_options(concepts, 7)
    positions = {
        concept: {permuted_options(concepts, seed).index(concept) for seed in range(20)}
        for concept in concepts
    }
    assert all(len(seen) > 1 for seen in positions.values())


def test_build_examples_derives_label_after_permutation() -> None:
    examples = ift.build_examples(
        cast(LoadedModel, _ChatOnly()),
        _bank(3, ("bread", "clock", "ocean", "spider")),
        [3],
        [0.2],
        seeds=range(8),
    )
    for example in examples:
        numbered = [
            line.split(". ", 1)[1]
            for line in example.prompt.splitlines()
            if line[:1].isdigit() and ". " in line
        ]
        assert numbered[example.concept_idx] == example.concept_name


def test_cross_layer_bank_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires a bank constructed at layer 2"):
        ift.build_examples(cast(LoadedModel, _ChatOnly()), _bank(3), [2], [0.2], seeds=[0])


def test_mismatched_ift_probe_plot_is_retired(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="retired"):
        plot_ift_vs_probe(tmp_path / "legacy.json", tmp_path / "plot.png")


def test_evaluate_disables_dropout_and_restores_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = SimpleNamespace(model=nn.Sequential(nn.Dropout(0.5)))
    wrapper.model.train()
    wrapper.chat = lambda user: user
    observed_modes: list[bool] = []

    def fake_score(*_args: object, **_kwargs: object) -> SimpleNamespace:
        observed_modes.append(wrapper.model.training)
        return SimpleNamespace(argmax=0)

    monkeypatch.setattr(ift, "score_choices", fake_score)
    assert ift.evaluate_layer(
        cast(LoadedModel, wrapper), _bank(2, ("ocean",)), 2, 0.2, seeds=[0]
    ) == [True]
    assert observed_modes == [False]
    assert wrapper.model.training is True


class _TrainBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x + self.bias,)


class _TrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TrainBlock(2)])
        self.readout = nn.Linear(2, 2)
        self.modes: list[bool] = []

    def forward(self, ids: torch.Tensor) -> SimpleNamespace:
        self.modes.append(self.training)
        hidden = torch.nn.functional.one_hot(ids, num_classes=2).float()
        for block in self.layers:
            hidden = block(hidden)[0]
        return SimpleNamespace(logits=self.readout(hidden))


class _TrainWrapper:
    def __init__(self) -> None:
        self.model = _TrainModel()
        self.device = torch.device("cpu")
        self.blocks = self.model.layers

    def encode(self, _prompt: str) -> torch.Tensor:
        return torch.tensor([[0]], dtype=torch.long)


def test_train_enables_dropout_mode_and_restores_eval_mode() -> None:
    wrapper = _TrainWrapper()
    wrapper.model.eval()
    example = ift.Example(
        prompt="prompt",
        concept_name="ocean",
        concept_idx=0,
        layer=0,
        strength=0.1,
        vector=torch.tensor([1.0, 0.0]),
    )
    losses = ift.train(
        cast(LoadedModel, wrapper), [example], [0, 1], epochs=1, lr=1e-3, log_every=0
    )
    assert len(losses) == 1
    assert wrapper.model.modes == [True]
    assert wrapper.model.training is False


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> str:
    text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _load_profile(path: Path, *, allow_legacy_schema1: bool = False) -> dict[int, float]:
    return load_matched_profile(
        path,
        expected_model="model/revision",
        expected_strength=0.2,
        expected_concepts=["bread", "ocean"],
        expected_n_layers=3,
        expected_model_revision="commit123",
        allow_legacy_schema1=allow_legacy_schema1,
    )


def test_matched_profile_loader_accepts_portable_v2_bundle(tmp_path: Path) -> None:
    summary_path = tmp_path / "profile.summary.json"
    raw_path = tmp_path / "profile.trials.jsonl"
    natural_path = tmp_path / "profile.natural-trials.jsonl"
    vector_hashes = {"bread": "a" * 64, "ocean": "b" * 64}
    raw_records = [
        {
            "injection_layer": 1,
            "concept": "bread",
            "concept_class": 0,
            "trial_id": 3,
            "probe_prediction_class": 0,
            "probe_correct": True,
            "probe_class_probabilities": [0.8, 0.2],
            "option_order": ["bread", "ocean"],
            "target_option_index": 0,
            "self_report_option_index": 0,
            "self_report_correct": True,
            "self_report_digit_logprobs": [-0.2, -1.6],
            "null_probe_prediction_classes": [1, 0],
            "null_probe_correct": [False, True],
            "rendered_prompt": "prompt bread",
            "rendered_prompt_sha256": hashlib.sha256(b"prompt bread").hexdigest(),
            "vector_sha256": vector_hashes["bread"],
            "prompt_variant": 3,
        },
        {
            "injection_layer": 1,
            "concept": "ocean",
            "concept_class": 1,
            "trial_id": 3,
            "probe_prediction_class": 0,
            "probe_correct": False,
            "probe_class_probabilities": [0.6, 0.4],
            "option_order": ["bread", "ocean"],
            "target_option_index": 1,
            "self_report_option_index": 0,
            "self_report_correct": False,
            "self_report_digit_logprobs": [-0.3, -1.4],
            "null_probe_prediction_classes": [1, 0],
            "null_probe_correct": [True, False],
            "rendered_prompt": "prompt ocean",
            "rendered_prompt_sha256": hashlib.sha256(b"prompt ocean").hexdigest(),
            "vector_sha256": vector_hashes["ocean"],
            "prompt_variant": 3,
        },
    ]
    natural_records = [
        {
            "concept": "bread",
            "concept_class": 0,
            "template_id": 0,
            "template": NATURAL_TEMPLATES[0],
            "rendered_text": NATURAL_TEMPLATES[0].format(concept="bread"),
            "grouped_cv_prediction_class": 0,
            "grouped_cv_correct": True,
        },
        {
            "concept": "ocean",
            "concept_class": 1,
            "template_id": 0,
            "template": NATURAL_TEMPLATES[0],
            "rendered_text": NATURAL_TEMPLATES[0].format(concept="ocean"),
            "grouped_cv_prediction_class": 0,
            "grouped_cv_correct": False,
        },
    ]
    raw_sha = _write_jsonl(raw_path, raw_records)
    natural_sha = _write_jsonl(natural_path, natural_records)
    summary = {
        "schema_version": 2,
        "estimand": MATCHED_ESTIMAND,
        "model": "model/revision",
        "model_revision": "commit123",
        "strength": 0.2,
        "concepts": ["bread", "ocean"],
        "n_model_layers": 3,
        "output_layer": 2,
        "null_permutations": 2,
        "raw_trials": raw_path.name,
        "raw_trials_sha256": raw_sha,
        "natural_trials": natural_path.name,
        "natural_trials_sha256": natural_sha,
        "prompt_provenance": {
            "identify_variant_templates": {
                str(index): IDENTIFY_FORCED_CHOICE_VARIANTS[index] for index in ift.EVAL_VARIANTS
            },
            "natural_templates": NATURAL_TEMPLATES,
            "concept_vector_templates": TEMPLATES,
            "concept_vector_neutral_fillers": NEUTRAL_FILLERS,
        },
        "provenance": {"source_sha256": {"runner.py": "c" * 64}},
        "readout": {"natural_train_n": 2, "natural_grouped_cv_accuracy": 0.5},
        "layers": [
            {
                "injection_layer": 1,
                "valid": True,
                "n_trials": 2,
                "probe_accuracy": 0.5,
                "self_report_accuracy": 0.5,
                "permuted_label_accuracy": 0.5,
                "permuted_label_accuracy_by_permutation": [0.5, 0.5],
                "bank_vector_sha256": vector_hashes,
            }
        ],
    }
    summary_path.write_text(json.dumps(summary))

    assert _load_profile(summary_path) == {1: 0.5}

    summary["layers"] = [
        {
            "injection_layer": 1,
            "valid": True,
            "n_trials": 2,
            "probe_accuracy": 0.5,
            "self_report_accuracy": 0.5,
            "permuted_label_accuracy": 0.25,
            "permuted_label_accuracy_by_permutation": [0.5, 0.5],
            "bank_vector_sha256": vector_hashes,
        }
    ]
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="permuted_label_accuracy disagrees"):
        _load_profile(summary_path)


def test_matched_profile_loader_keeps_schema1_path_compatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    bundle = tmp_path / "results"
    bundle.mkdir()
    summary_path = bundle / "legacy.summary.json"
    raw_path = bundle / "legacy.trials.jsonl"
    raw_sha = _write_jsonl(
        raw_path,
        [{"injection_layer": 1}, {"injection_layer": 1}],
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "estimand": MATCHED_ESTIMAND,
                "model": "model/revision",
                "model_revision": "commit123",
                "strength": 0.2,
                "concepts": ["bread", "ocean"],
                "n_model_layers": 3,
                "output_layer": 2,
                # Schema 1 wrote paths relative to the process cwd.
                "raw_trials": "results/legacy.trials.jsonl",
                "raw_trials_sha256": raw_sha,
                "layers": [
                    {
                        "injection_layer": 1,
                        "valid": True,
                        "n_trials": 2,
                        "probe_accuracy": 0.5,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="legacy schema_version 1"):
        _load_profile(summary_path)
    assert _load_profile(summary_path, allow_legacy_schema1=True) == {1: 0.5}


def test_matched_profile_loader_rejects_checksum_or_raw_summary_tampering(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "legacy.summary.json"
    raw_path = tmp_path / "legacy.trials.jsonl"
    raw_sha = _write_jsonl(raw_path, [{"injection_layer": 1}])
    payload = {
        "schema_version": 1,
        "estimand": MATCHED_ESTIMAND,
        "model": "model/revision",
        "model_revision": "commit123",
        "strength": 0.2,
        "concepts": ["bread", "ocean"],
        "n_model_layers": 3,
        "output_layer": 2,
        "raw_trials": raw_path.name,
        "raw_trials_sha256": raw_sha,
        "layers": [{"injection_layer": 1, "valid": True, "n_trials": 1, "probe_accuracy": 1.0}],
    }
    summary_path.write_text(json.dumps(payload))
    raw_path.write_text('{"injection_layer": 1, "tampered": true}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        _load_profile(summary_path, allow_legacy_schema1=True)

    repaired_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    payload["raw_trials_sha256"] = repaired_sha
    payload["layers"] = [
        {"injection_layer": 1, "valid": True, "n_trials": 2, "probe_accuracy": 1.0}
    ]
    summary_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="raw trial count mismatch"):
        _load_profile(summary_path, allow_legacy_schema1=True)
