"""Cross-DEV query-head screen for the four frozen Stage 1a layer/role pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_attention_localization as stage1  # noqa: E402

from introspect import concepts, models, retained  # noqa: E402
from introspect.attention_patching import patch_attention_inputs  # noqa: E402
from introspect.codebook_icl import (  # noqa: E402
    LABELS,
    VISIBLE_SAMPLES,
    Episode,
    PreparedEpisode,
    condition_interventions,
    exact_episodes,
    prepare_episode,
    sha256_text,
    tensor_sha256,
)
from introspect.concepts import ConceptVector  # noqa: E402
from introspect.hooks import intervene  # noqa: E402
from introspect.models import LoadedModel  # noqa: E402

CONCEPTS = ("bread", "volcano", "violin")
CARRIER_INDICES = (1, 2)
CARRIERS = tuple(VISIBLE_SAMPLES[index] for index in CARRIER_INDICES)
PAIRS = (
    (21, "query_marker"),
    (23, "query_marker"),
    (26, "final_answer"),
    (31, "final_answer"),
)
HEADS = tuple(range(16))
SENTINELS = (7, 8, 9)
_SOURCE_PATHS = tuple(
    dict.fromkeys((*stage1._SOURCE_PATHS, "scripts/run_attention_head_screen.py"))
)


def complementary_dev_episodes(sample: str) -> list[Episode]:
    """The opposite mapping half from Stage 1a, with both query-sign twins."""
    episodes = exact_episodes(sample)
    orders = sorted({episode.demo_signs for episode in episodes})
    selected = [
        episode
        for order_id, order in enumerate(orders)
        for episode in episodes
        if episode.demo_signs == order and episode.positive_label == LABELS[1 - order_id % 2]
    ]
    stage1_ids = {episode.cell_id for episode in stage1.balanced_dev_episodes(sample)}
    if (
        len(selected) != 12
        or len({episode.demo_signs for episode in selected}) != 6
        or sum(episode.query_sign == 1 for episode in selected) != 6
        or sum(episode.positive_label == "Q" for episode in selected) != 6
        or sum(episode.correct_label == "Q" for episode in selected) != 6
        or stage1_ids.intersection(episode.cell_id for episode in selected)
    ):
        raise RuntimeError("the complementary DEV subset lost balance or overlaps Stage 1a")
    return selected


def _pair_id(layer: int, role: str) -> str:
    return f"{role}@{layer}"


def _episodes(sample: str, smoke: bool) -> list[Episode]:
    episodes = complementary_dev_episodes(sample)
    return episodes[:2] if smoke else episodes


def _run_axes(
    smoke: bool,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[int, str], ...],
    tuple[int, ...],
    tuple[tuple[int, str], ...],
    list[Episode],
]:
    concepts_used = CONCEPTS[:1] if smoke else CONCEPTS
    carriers_used = (
        ((CARRIER_INDICES[0], CARRIERS[0]),)
        if smoke
        else tuple(zip(CARRIER_INDICES, CARRIERS, strict=True))
    )
    pairs_used = PAIRS[:1] if smoke else PAIRS
    heads_used = HEADS[:2] if smoke else HEADS
    episodes = _episodes(carriers_used[0][1], smoke)
    return concepts_used, carriers_used, heads_used, pairs_used, episodes


def expected_scored_forwards(smoke: bool) -> int:
    concepts_used, carriers_used, heads_used, pairs_used, episodes = _run_axes(smoke)
    units = len(concepts_used) * len(carriers_used) * len(episodes)
    return units * (3 + len(pairs_used) * (1 + len(heads_used)))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    return {path: _sha256(root / path) for path in _SOURCE_PATHS}


def _publish_no_overwrite(temporary: Path, destination: Path) -> None:
    """Make a completed temporary file visible without clobbering a late writer."""
    os.link(temporary, destination)
    temporary.unlink()


def _write_json_no_overwrite(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    created = False
    try:
        with temporary.open("x") as handle:
            created = True
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _publish_no_overwrite(temporary, path)
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise


def build_protocol(source_hashes: dict[str, str], *, frozen_on: str) -> dict[str, Any]:
    """Build the exact protocol candidate; writing/freezing it is a separate review step."""
    date.fromisoformat(frozen_on)
    return {
        "schema_version": 1,
        "frozen_on": frozen_on,
        "status": "DEV_ONLY_CROSS_DEV_HEAD_SCREEN_NOT_CONFIRMATORY",
        "safety_goal": (
            "Test whether the four selected Stage 1a layer/receiver effects localize to at "
            "most four cross-DEV-stable query-head contributions spanning query integration "
            "and final reporting."
        ),
        "design": {
            "model": stage1.MODEL,
            "model_revision": stage1.MODEL_REVISION,
            "device": "cpu",
            "dtype": "float32",
            "concepts": list(CONCEPTS),
            "carrier_indices": list(CARRIER_INDICES),
            "carriers": list(CARRIERS),
            "cells": (
                "the complementary mapping for all six demonstration orders, with both "
                "query-sign twins (12 cells per concept/carrier)"
            ),
            "units": 72,
            "injection_layer": stage1.INJECTION_LAYER,
            "strength": stage1.STRENGTH,
            "pre_injection_sentinel_layers": list(SENTINELS),
            "selected_layer_roles": [
                {"layer": layer, "role": role, "id": _pair_id(layer, role)} for layer, role in PAIRS
            ],
            "query_heads": list(HEADS),
            "n_query_heads": 16,
            "n_kv_heads": 2,
            "head_width": 128,
            "candidate_components": 64,
            "candidate_identity": "distinct (layer, receiver role, query head) triples",
            "arms_per_unit": 71,
            "scored_forwards": 5112,
            "interchange": (
                "replace either all 16 or one pre-o_proj query-head context slice at the "
                "selected receiver positions with the exact paired test_only context"
            ),
        },
        "instrumentation_gates": {
            "forward_mode": "eval, no_grad, use_cache=False, CPU float32",
            "model_loading": "pinned revision from the project-local cache in offline mode",
            "pre_injection_sentinels": "target and test_only must be bit-identical",
            "self_donor": "selected-layer all-position target self-patch must be bit-exact",
            "recipient_binding": "each patched pre-o_proj input must equal its cached target",
            "format_binding": "save the full-vocabulary argmax token ID and the Q/K token IDs",
            "artifact_binding": (
                "save prompt, token, direction, donor, recipient, and full-logit hashes"
            ),
        },
        "analysis_rules": {
            "unit": (
                "concept and carrier are crossed development strata; the 12 exact episode "
                "cells are nuisance marginalization"
            ),
            "aggregation": (
                "compute every stratum statistic as an unweighted mean over its 12 cells, "
                "then compute every aggregate statistic as an unweighted mean over the six "
                "concept-carrier strata; accuracy and format are means of their booleans"
            ),
            "removal_fraction": (
                "mean_strata(mean_cells(target signed margin - patch signed margin)) divided "
                "by mean_strata(mean_cells(target signed margin - test_only signed margin)); "
                "never average rowwise ratios, and the aggregate denominator must be positive"
            ),
            "label_mass_retention": (
                "mean_strata(mean_cells(patched label mass)) divided by "
                "mean_strata(mean_cells(target label mass)); the ratio must be >= 0.80"
            ),
            "baseline_gate": (
                "aggregate target-test_only conditional-correct-probability gap >= 0.15, "
                "accuracy gap >= 0.25, aggregate signed-margin gap > 0, and a positive "
                "mean signed-margin gap in >= 5/6 concept-carrier strata"
            ),
            "layer_replication_gate": (
                "all four selected all-head parent patches must each remove >= 20% of the "
                "aggregate baseline margin, retain format >= 0.90 and label mass >= 80%, "
                "and have a positive margin drop in >= 5/6 concept-carrier strata"
            ),
            "head_candidate_gate": (
                "a single query head must remove >= 10% of the aggregate baseline margin, "
                "retain format >= 0.90 and label mass >= 80%, and have a positive margin "
                "drop in >= 5/6 concept-carrier strata"
            ),
            "selection": (
                "evaluate all 64 distinct layer-role-head components and retain every passer; "
                "there is no top-k truncation"
            ),
            "sparse_go": (
                "proceed only if 2-4 layer-role-head components qualify in total and both "
                "query_marker and final_answer receiver families are represented"
            ),
            "stop": (
                "stop the single-route study if the baseline fails, any of the four parent "
                "patches fails, fewer than two or more than four components qualify, or "
                "either receiver family is absent"
            ),
            "multiplicity": (
                "development selection only: no p-values, population intervals, or additive "
                "interpretation of overlapping layer-role-head effects"
            ),
            "hard_handoff": (
                "a sparse go authorizes only a separately frozen Stage 1c joint/envelope "
                "decomposition of all retained components against their all-head parents "
                "and the all-positions envelope; it does not authorize a mechanism or safety "
                "claim"
            ),
        },
        "claim_boundary": (
            "A positive screen identifies paired-interchange-sensitive query-head "
            "contributions on cross-DEV prompts. It does not establish QK-only mediation, "
            "individual or joint necessity, a readable program, semantic introspection, or "
            "safety-monitor robustness."
        ),
        "smoke_disclosure": (
            "Smoke may use one query-twin pair, the all-16-head query_marker@21 parent, "
            "and individual heads 0/1 only for wiring. It is never selection evidence "
            "and may not alter pairs, heads, or gates."
        ),
        "previous_protocol_disclosure": (
            "Protocol v1 was frozen but never run. Under V2, one smoke unit completed in "
            "memory before the final compute-budget assertion caught a stale one-cell slice; "
            "the runner deleted its temporary file, wrote no raw or manifest artifact, and "
            "printed no result value. V3 removes that duplicate slice so the already frozen "
            "query-twin smoke executes both cells; the full design and analysis are unchanged."
        ),
        "stage1a_protocol_sha256": (
            "27c8af5f4917dc3c72214caece71ac996d9c26ec914dd390f86f954a73e41427"
        ),
        "stage1a_raw_sha256": ("530f4f550e514cb64787d3b8206742533f2e70eab1839ed018e9c24bd84d5c1c"),
        "stage1a_analyzer_sha256": (
            "025a0addecc93b110d56e8123a8e75ac91681baaf83f87ca9cbbd105b2fa6d2c"
        ),
        "source_files_sha256": source_hashes,
    }


def load_protocol(path: Path, source_hashes: dict[str, str]) -> tuple[dict[str, Any], str]:
    raw = path.read_text()
    protocol = cast(dict[str, Any], json.loads(raw))
    frozen_on = protocol.get("frozen_on")
    if not isinstance(frozen_on, str):
        raise ValueError("protocol must include a frozen_on date")
    if protocol != build_protocol(source_hashes, frozen_on=frozen_on):
        raise ValueError("protocol does not match the executable design and source hashes")
    return protocol, sha256_text(raw)


def _add_argmax(
    score: dict[str, object], logits: Tensor, prepared: PreparedEpisode
) -> dict[str, object]:
    argmax = int(logits.argmax())
    if score["format_ok"] != (argmax in set(prepared.label_ids)):
        raise RuntimeError("saved full-vocabulary argmax disagrees with format_ok")
    score["full_argmax_token_id"] = argmax
    return score


@torch.no_grad()
def _patched_score(
    model: LoadedModel,
    prepared: PreparedEpisode,
    direction: ConceptVector,
    *,
    layer: int,
    role: stage1.ReceiverPositions,
    heads: Sequence[int],
    n_heads: int,
    donor: Tensor,
    expected_recipient: Tensor,
    target_logits: Tensor,
) -> dict[str, object]:
    interventions = condition_interventions(
        "target",
        direction,
        prepared.state_positions,
        prepared.episode.state_signs,
        strength=stage1.STRENGTH,
    )
    with (
        intervene(model, interventions, prompt_len=int(prepared.input_ids.shape[1])),
        patch_attention_inputs(
            model,
            {layer: donor},
            {layer: heads},
            n_heads=n_heads,
            expected_recipients={layer: expected_recipient},
            positions=role,
        ),
    ):
        logits = stage1._forward_logits(model, prepared.input_ids)[0, -1].float()
    score = stage1._score(logits, prepared)
    score["target_to_patched_kl_nats"] = stage1._kl_nats(target_logits, logits)
    return _add_argmax(score, logits, prepared)


def _directions(model: LoadedModel) -> tuple[dict[str, ConceptVector], Tensor, float]:
    raw_bank = concepts.build_bank(
        model, stage1.INJECTION_LAYER, list(retained.DEV_CONCEPTS), center=False
    )
    center = torch.stack([item.vector for item in raw_bank.values()]).mean(0)
    bank = {
        name: ConceptVector(name, stage1.INJECTION_LAYER, item.vector - center)
        for name, item in raw_bank.items()
    }
    max_cosine = concepts.max_offdiagonal_cosine(bank)
    if max_cosine > 0.5:
        raise RuntimeError(f"centered DEV directions are near-collinear ({max_cosine:.3f})")
    return {name: bank[name] for name in CONCEPTS}, center, max_cosine


def run(*, out: Path, protocol_path: Path, smoke: bool) -> None:
    manifest_path = out.with_suffix(".manifest.json")
    temporary = out.with_name(f".{out.name}.tmp")
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    if out.exists() or manifest_path.exists() or temporary.exists() or manifest_temporary.exists():
        raise SystemExit("refusing to overwrite an existing raw, manifest, or temporary artifact")

    source_hashes = _source_hashes(ROOT)
    protocol, protocol_sha = load_protocol(protocol_path, source_hashes)
    concepts_used, carriers_used, heads_used, pairs_used, episodes = _run_axes(smoke)
    capture_layers = SENTINELS + tuple(sorted({layer for layer, _role in pairs_used}))
    torch.manual_seed(0)
    model = models.load(stage1.MODEL, device=torch.device("cpu"), revision=stage1.MODEL_REVISION)
    try:
        if model.n_layers != 36 or models.loaded_revision(model) != stage1.MODEL_REVISION:
            raise RuntimeError("the loaded model or revision differs from the frozen design")
        if model.device != torch.device("cpu") or model.dtype != torch.float32:
            raise RuntimeError("head screening is frozen to CPU float32")
        if cast(Any, model.model).training:
            raise RuntimeError("head screening requires eval mode with dropout disabled")
        model_config = cast(Any, model.model).config
        model_config.use_cache = False
        if model_config.use_cache is not False:
            raise RuntimeError("head screening could not disable the model KV cache")
        n_heads = int(model_config.num_attention_heads)
        n_kv_heads = int(model_config.num_key_value_heads)
        if n_heads != 16 or n_kv_heads != 2 or model.d_model != 2048:
            raise RuntimeError("loaded Qwen head layout differs from the frozen design")

        all_directions, center, max_cosine = _directions(model)
        directions = {name: all_directions[name] for name in concepts_used}
        prepared_by_carrier: dict[int, list[PreparedEpisode]] = {}
        labels_by_unit: dict[tuple[int, str], tuple[int, ...]] = {}
        roles_by_unit: dict[tuple[int, str], dict[str, stage1.ReceiverPositions]] = {}
        for carrier_index, sample in carriers_used:
            carrier_episodes = _episodes(sample, smoke)
            prepared_by_carrier[carrier_index] = [
                prepare_episode(model, episode) for episode in carrier_episodes
            ]
            for prepared in prepared_by_carrier[carrier_index]:
                key = (carrier_index, prepared.episode.cell_id)
                labels = stage1._demo_label_positions(model, prepared)
                labels_by_unit[key] = labels
                roles_by_unit[key] = stage1.receiver_roles(prepared, labels)

        prompt_hashes = [
            prepared.prompt_sha256
            for _concept in concepts_used
            for carrier_index, _sample in carriers_used
            for prepared in prepared_by_carrier[carrier_index]
        ]
        config: dict[str, object] = {
            "schema_version": 1,
            "status": (
                "SMOKE_INSTRUMENTATION_ONLY"
                if smoke
                else "DEV_ONLY_CROSS_DEV_HEAD_SCREEN_NOT_CONFIRMATORY"
            ),
            "estimand": "cross-DEV single-query-head paired-interchange sensitivity",
            "model_requested": stage1.MODEL,
            "model_resolved": model.name,
            "model_revision": stage1.MODEL_REVISION,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "concepts": list(concepts_used),
            "carriers": [sample for _index, sample in carriers_used],
            "carrier_indices": [index for index, _sample in carriers_used],
            "cell_ids": [episode.cell_id for episode in episodes],
            "injection_layer": stage1.INJECTION_LAYER,
            "strength": stage1.STRENGTH,
            "pre_injection_sentinel_layers": list(SENTINELS),
            "selected_layer_roles": [
                {"layer": layer, "role": role, "id": _pair_id(layer, role)}
                for layer, role in pairs_used
            ],
            "heads": list(heads_used),
            "n_query_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_width": model.d_model // n_heads,
            "candidate_components": len(pairs_used) * len(heads_used),
            "scored_forwards_expected": expected_scored_forwards(smoke),
            "forward_mode": "eval, no_grad, use_cache=False, CPU float32",
            "offline_model_loading": True,
            "balanced_subset": (
                "one complementary DEV cell (instrumentation only)"
                if smoke
                else "complementary mapping; six orders; both query-sign twins"
            ),
            "direction_sha256": {
                name: tensor_sha256(direction.vector) for name, direction in directions.items()
            },
            "centering_direction_sha256": tensor_sha256(center),
            "centering_concepts": list(retained.DEV_CONCEPTS),
            "max_offdiagonal_cosine": max_cosine,
            "prompt_set_sha256": stage1._json_sha256(sorted(prompt_hashes)),
            "source_files_sha256": source_hashes,
            "source_sha256": stage1._json_sha256(source_hashes),
            "protocol": protocol,
            "protocol_sha256": protocol_sha,
            "smoke": smoke,
            "selection_eligible": not smoke,
            "publishable": False,
            "git_commit": stage1._git_commit(ROOT),
            "git_dirty": stage1._git_dirty(ROOT),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        }
        config_sha = stage1._json_sha256(config)
        expected_units = len(concepts_used) * len(carriers_used) * len(episodes)
        started = time.time()
        rows = 0
        created_temporary = False
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("x") as handle:
                created_temporary = True
                for concept in concepts_used:
                    direction = directions[concept]
                    for carrier_id, (carrier_index, _sample) in enumerate(carriers_used):
                        for prepared in prepared_by_carrier[carrier_index]:
                            key = (carrier_index, prepared.episode.cell_id)
                            target, target_logits, target_contexts = stage1._capture_condition(
                                model, prepared, "target", direction, capture_layers
                            )
                            target["target_to_target_kl_nats"] = 0.0
                            _add_argmax(target, target_logits, prepared)
                            test_only, test_logits, donor_contexts = stage1._capture_condition(
                                model, prepared, "test_only", direction, capture_layers
                            )
                            test_only["target_to_test_only_kl_nats"] = stage1._kl_nats(
                                target_logits, test_logits
                            )
                            _add_argmax(test_only, test_logits, prepared)
                            for layer in SENTINELS:
                                if not torch.equal(target_contexts[layer], donor_contexts[layer]):
                                    raise RuntimeError(
                                        "target/test_only diverged before injection at "
                                        f"layer {layer}"
                                    )

                            pair_layers = tuple(sorted({layer for layer, _role in pairs_used}))
                            self_score, self_logits = stage1._self_patch(
                                model,
                                prepared,
                                direction,
                                layers=pair_layers,
                                n_heads=n_heads,
                                target_contexts={
                                    layer: target_contexts[layer] for layer in pair_layers
                                },
                            )
                            if not torch.equal(target_logits, self_logits):
                                raise RuntimeError("target self-donor sham changed full logits")
                            self_score["target_to_self_patch_kl_nats"] = stage1._kl_nats(
                                target_logits, self_logits
                            )
                            _add_argmax(self_score, self_logits, prepared)

                            pair_results: dict[str, object] = {}
                            for layer, role in pairs_used:
                                positions = roles_by_unit[key][role]
                                pair_results[_pair_id(layer, role)] = {
                                    "layer": layer,
                                    "role": role,
                                    "all_heads": _patched_score(
                                        model,
                                        prepared,
                                        direction,
                                        layer=layer,
                                        role=positions,
                                        heads=HEADS,
                                        n_heads=n_heads,
                                        donor=donor_contexts[layer],
                                        expected_recipient=target_contexts[layer],
                                        target_logits=target_logits,
                                    ),
                                    "heads": {
                                        str(head): _patched_score(
                                            model,
                                            prepared,
                                            direction,
                                            layer=layer,
                                            role=positions,
                                            heads=(head,),
                                            n_heads=n_heads,
                                            donor=donor_contexts[layer],
                                            expected_recipient=target_contexts[layer],
                                            target_logits=target_logits,
                                        )
                                        for head in heads_used
                                    },
                                }

                            token_ids = prepared.input_ids[0].tolist()
                            row = {
                                "schema_version": 1,
                                "config_sha256": config_sha,
                                "unit_id": (
                                    f"{concept}/carrier-{carrier_index}/{prepared.episode.cell_id}"
                                ),
                                "concept": concept,
                                "carrier_id": carrier_id,
                                "source_carrier_index": carrier_index,
                                "cell_id": prepared.episode.cell_id,
                                "episode_sha256": prepared.episode.digest(),
                                "prompt": prepared.prompt,
                                "prompt_sha256": prepared.prompt_sha256,
                                "token_ids": token_ids,
                                "token_ids_sha256": stage1._json_sha256(token_ids),
                                "label_token_ids": dict(
                                    zip(LABELS, prepared.label_ids, strict=True)
                                ),
                                "state_token_positions": prepared.state_positions,
                                "query_marker_position": prepared.state_positions[-1],
                                "demo_label_positions": labels_by_unit[key],
                                "final_answer_position": int(prepared.input_ids.shape[1]) - 1,
                                "demo_signs": prepared.episode.demo_signs,
                                "query_sign": prepared.episode.query_sign,
                                "label_mapping": {
                                    "+1": prepared.episode.positive_label,
                                    "-1": prepared.episode.negative_label,
                                },
                                "correct_label": prepared.episode.correct_label,
                                "direction_sha256": tensor_sha256(direction.vector),
                                "baseline": {"target": target, "test_only": test_only},
                                "instrumentation": {
                                    "pre_injection_sentinels_equal": True,
                                    "self_patch_score": self_score,
                                    "self_patch_exact": True,
                                    "self_patch_max_abs_logit_error": 0.0,
                                    "attention_context_sha256": {
                                        "target": {
                                            str(layer): tensor_sha256(target_contexts[layer])
                                            for layer in capture_layers
                                        },
                                        "test_only": {
                                            str(layer): tensor_sha256(donor_contexts[layer])
                                            for layer in capture_layers
                                        },
                                    },
                                },
                                "pairs": pair_results,
                            }
                            handle.write(json.dumps(row, sort_keys=True) + "\n")
                            handle.flush()
                            rows += 1
                            print(
                                f"unit {rows}/{expected_units} {row['unit_id']} "
                                f"[{time.time() - started:.0f}s]",
                                flush=True,
                            )
            if source_hashes != _source_hashes(ROOT):
                raise RuntimeError("generation source changed while the run was active")
            n_forwards = rows * (3 + len(pairs_used) * (1 + len(heads_used)))
            if rows != expected_units or n_forwards != expected_scored_forwards(smoke):
                raise RuntimeError("completed artifact does not match its frozen compute budget")
            raw_sha = _sha256(temporary)
            _publish_no_overwrite(temporary, out)
        except BaseException:
            if created_temporary:
                temporary.unlink(missing_ok=True)
            raise

        manifest = {
            "schema_version": 1,
            "config": config,
            "config_sha256": config_sha,
            "raw": out.name,
            "raw_sha256": raw_sha,
            "n_unit_rows": rows,
            "n_scored_forwards": n_forwards,
        }
        _write_json_no_overwrite(manifest_path, manifest)
        print(f"wrote {out} (sha256={raw_sha}, scored_forwards={n_forwards})")
        print(f"manifest {manifest_path}")
    finally:
        model.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(out=args.out, protocol_path=args.protocol, smoke=args.smoke)


if __name__ == "__main__":
    main()
