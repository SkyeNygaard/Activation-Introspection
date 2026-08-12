"""Does the model beat an equal-or-lower-cost third party reading the same states?

[Privileged Self-Access Matters](https://arxiv.org/abs/2508.14802) defines
introspection as a process yielding information about internal states "more
reliable than one with equal or lower computational cost available to a third
party". The confirmed causal-codebook result (0.891 against an exactly matched
0.500 query-only arm) has never been measured against that criterion, and
``spar-application/PROJECT-BRIEFS.md`` says so: "A classifier handed the same
retained state would plausibly do as well, and that control has not been run."

This runs it. On the identical frozen episodes, the same forward pass that scores
the model also captures the five post-injection residual states at the injection
site. A four-shot reader is then fitted on the four demonstration states and their
labels and asked for the fifth. The reader is strictly cheaper than the model: it
performs O(n*d) arithmetic on five 2048-vectors, while the model continues through
27 further transformer blocks from the same site.

The comparison is paired by episode, so it is the same 576 units either way.
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

from introspect import concepts as concept_mod
from introspect import models, retained
from introspect.codebook_icl import (
    CONFIRM_CONCEPTS,
    CONFIRM_VISIBLE_SAMPLES,
    LABELS,
    PreparedEpisode,
    condition_interventions,
    exact_episodes,
    prepare_episode,
    sha256_text,
)
from introspect.hooks import capture, intervene
from introspect.preflight import check as preflight_check

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LAYER = 9
STRENGTH = 1.0
#: The frozen number this run has to reproduce before its comparison means anything.
FROZEN_TARGET_ACCURACY = 0.891
REPRODUCTION_TOLERANCE = 0.05
READERS = ("centroid_euclidean", "centroid_cosine", "shuffled_labels")
SOURCE_PATHS = (
    "scripts/run_matched_reader.py",
    "src/introspect/codebook_icl.py",
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
        "disclosed_precursors": (
            "v1 was frozen and produced no raw artifact: the smoke crashed in the "
            "per-concept aggregation, dividing by an empty subset for concepts the "
            "smoke never ran. v2 ran a six-episode smoke in which the shuffled-label "
            "reader scored 1.000 — an implementation defect, not a result: the "
            "permutation was seeded by position within a carrier, so only two draws "
            "were used, and a permutation preserving the balanced 2/2 grouping leaves "
            "the reader's centroids unchanged. v3 seeds one permutation per episode. "
            "The design, criterion, gates and pre-registered interpretation are "
            "unchanged throughout; the model's own arm was never inspected before this "
            "freeze beyond a 6-episode 1.000 that the reproduction gate rejects. "
            "Expected shuffled accuracy is about 0.58, not 0.50, because one draw in "
            "six preserves the grouping; the frozen band admits this."
        ),
        "question": (
            "On the episodes that produced 0.891, does the model's in-context report "
            "exceed a four-shot reader given the same five residual states at the same "
            "site?"
        ),
        "criterion": (
            "arXiv 2508.14802: introspection requires being more reliable than a process "
            "of equal or lower computational cost available to a third party. The reader "
            "here is strictly lower cost: O(n*d) arithmetic on five 2048-vectors against "
            "27 remaining transformer blocks."
        ),
        "prereg_interpretation": {
            "reader_at_or_above_model": (
                "no privileged access at this site under the stated criterion. The 0.891 "
                "is evidence that a causally injected hidden state is usable as an "
                "in-context channel, NOT that the model reads it better than an outside "
                "reader can. This is the expected outcome and is not a failed run."
            ),
            "model_above_reader": (
                "the model extracts something the cheapest matched reader does not. "
                "Report the paired margin; do not call it privileged access without "
                "also ruling out that a slightly richer reader closes the gap."
            ),
            "note": (
                "written before the run precisely because the expected direction is "
                "unflattering to the headline number"
            ),
        },
        "reproduction_gate": {
            "frozen_target_accuracy": FROZEN_TARGET_ACCURACY,
            "tolerance": REPRODUCTION_TOLERANCE,
            "rule": (
                "the re-scored target arm must land within tolerance of the frozen "
                "confirmation, or this is not the same experiment and the comparison "
                "is void"
            ),
        },
        "reader_sanity_gate": {
            "shuffled_labels_accuracy_band": [0.35, 0.65],
            "rule": (
                "permuting the four demonstration labels must collapse the reader to "
                "chance; otherwise the reader is exploiting something other than the "
                "labelled states"
            ),
        },
        "design": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "strength": STRENGTH,
            "concepts": list(CONFIRM_CONCEPTS),
            "visible_samples": list(CONFIRM_VISIBLE_SAMPLES),
            "labels": list(LABELS),
            "readers": list(READERS),
            "reader_information": (
                "the five residual states captured at the injection site in the same "
                "forward pass that scored the model, plus the four demonstration labels"
            ),
            "unit": "episode; paired model-versus-reader on identical episodes",
            "centering": "DEV-concept mean subtracted, exactly as the frozen confirmation",
        },
        "known_asymmetries": [
            "the model must also parse the prompt, apply the episode's remapped label "
            "convention, and emit a correctly formatted token; the reader does none of "
            "these and is scored only on the state-to-label decision",
            "the reader sees the post-injection state directly, which is the access a "
            "third party is stipulated to have under the criterion, not an unfair extra",
            "a reader this cheap cannot be beaten by making it richer, so a model win "
            "would need the opposite check: that richer readers do not close the gap",
        ],
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


def _read(states: Tensor, signs: tuple[int, ...], reader: str, seed: int) -> int:
    """Predict the query state's sign from four labelled demonstration states.

    ``states`` is [5, d]: four demonstrations then the query. Nearest centroid is
    the cheapest reader that can use the labels at all, which is what the
    equal-or-lower-cost criterion asks for.
    """
    demos, query = states[:4], states[4]
    demo_signs = list(signs[:4])
    if reader == "shuffled_labels":
        generator = torch.Generator().manual_seed(seed)
        demo_signs = [demo_signs[i] for i in torch.randperm(4, generator=generator).tolist()]

    positive = demos[[i for i, sign in enumerate(demo_signs) if sign == 1]].mean(0)
    negative = demos[[i for i, sign in enumerate(demo_signs) if sign == -1]].mean(0)
    if reader == "centroid_cosine":
        similarity = torch.nn.functional.cosine_similarity(
            torch.stack([positive, negative]), query.unsqueeze(0), dim=1
        )
        return 1 if float(similarity[0]) >= float(similarity[1]) else -1
    distances = torch.stack([positive - query, negative - query]).norm(dim=1)
    return 1 if float(distances[0]) <= float(distances[1]) else -1


@torch.no_grad()
def _episode_row(
    model: models.LoadedModel,
    prepared: PreparedEpisode,
    direction: concept_mod.ConceptVector,
    concept: str,
    carrier: int,
    seed: int,
) -> dict[str, object]:
    """One forward pass scores the model and supplies the reader's input."""
    episode = prepared.episode
    interventions = condition_interventions(
        "target",
        direction,
        prepared.state_positions,
        episode.state_signs,
        strength=STRENGTH,
    )
    with (
        intervene(model, interventions, prompt_len=int(prepared.input_ids.shape[1])),
        capture(model, [LAYER]) as store,
    ):
        logits = model.forward_logits(prepared.input_ids)[0, -1].float().cpu()
    states = store.acts[LAYER][0][0, list(prepared.state_positions)].clone()

    candidates = logits[torch.tensor(prepared.label_ids)]
    model_label = LABELS[int(candidates.argmax())]
    row: dict[str, object] = {
        "concept": concept,
        "carrier": carrier,
        "cell_id": episode.cell_id,
        "demo_signs": list(episode.demo_signs),
        "query_sign": episode.query_sign,
        "positive_label": episode.positive_label,
        "correct_label": episode.correct_label,
        "prompt_sha256": prepared.prompt_sha256,
        "state_positions": list(prepared.state_positions),
        "model_predicted": model_label,
        "model_correct": model_label == episode.correct_label,
        "model_format_ok": int(logits.argmax()) in set(prepared.label_ids),
        "state_norms": [float(state.norm()) for state in states],
    }
    for reader in READERS:
        sign = _read(states, episode.state_signs, reader, seed)
        label = episode.label_for(sign)
        row[f"reader_{reader}_predicted"] = label
        row[f"reader_{reader}_correct"] = label == episode.correct_label
    return row


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    model_accuracy = sum(bool(row["model_correct"]) for row in rows) / total
    readers = {
        reader: sum(bool(row[f"reader_{reader}_correct"]) for row in rows) / total
        for reader in READERS
    }
    primary = "centroid_euclidean"
    both = sum(
        bool(row["model_correct"]) and bool(row[f"reader_{primary}_correct"]) for row in rows
    )
    model_only = sum(
        bool(row["model_correct"]) and not bool(row[f"reader_{primary}_correct"]) for row in rows
    )
    reader_only = sum(
        (not bool(row["model_correct"])) and bool(row[f"reader_{primary}_correct"]) for row in rows
    )
    neither = total - both - model_only - reader_only

    by_concept = {}
    for concept in dict.fromkeys(str(row["concept"]) for row in rows):
        subset = [row for row in rows if row["concept"] == concept]
        by_concept[concept] = {
            "model": sum(bool(row["model_correct"]) for row in subset) / len(subset),
            "reader": sum(bool(row[f"reader_{primary}_correct"]) for row in subset) / len(subset),
        }

    reproduced = abs(model_accuracy - FROZEN_TARGET_ACCURACY) <= REPRODUCTION_TOLERANCE
    shuffled_sane = 0.35 <= readers["shuffled_labels"] <= 0.65
    return {
        "n_episodes": total,
        "model_target_accuracy": model_accuracy,
        "reader_accuracy": readers,
        "model_minus_reader": model_accuracy - readers[primary],
        "paired_counts": {
            "both_correct": both,
            "model_only": model_only,
            "reader_only": reader_only,
            "neither": neither,
        },
        "by_concept": by_concept,
        "model_format_rate": sum(bool(row["model_format_ok"]) for row in rows) / total,
        "gates": {
            "reproduces_frozen_confirmation": reproduced,
            "shuffled_label_reader_at_chance": shuffled_sane,
        },
        "verdict": (
            "void_did_not_reproduce"
            if not reproduced
            else "void_reader_control_failed"
            if not shuffled_sane
            else "no_privileged_access_at_this_site"
            if readers[primary] >= model_accuracy
            else "model_exceeds_matched_reader"
        ),
    }


def run(args: argparse.Namespace) -> None:
    out = args.out
    manifest_path = out.with_suffix(".manifest.json")
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    for path in (out, manifest_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    protocol, protocol_sha = _freeze_protocol(
        args.protocol or Path("results/matched_reader_protocol_v1.json")
    )
    preflight_check(MODEL, training=False)

    model = models.load(MODEL, revision=MODEL_REVISION)
    started = time.time()
    rows: list[dict[str, object]] = []
    try:
        if models.loaded_revision(model) != MODEL_REVISION:
            raise SystemExit("loaded model revision does not match the frozen revision")
        reference = concept_mod.build_bank(model, LAYER, list(retained.DEV_CONCEPTS), center=False)
        center = torch.stack([d.vector for d in reference.values()]).mean(0)
        raw_bank = concept_mod.build_bank(model, LAYER, list(CONFIRM_CONCEPTS), center=False)
        bank = {
            name: concept_mod.ConceptVector(name=name, layer=LAYER, vector=d.vector - center)
            for name, d in raw_bank.items()
        }

        concepts = CONFIRM_CONCEPTS[:1] if args.smoke else CONFIRM_CONCEPTS
        for carrier, sample in enumerate(CONFIRM_VISIBLE_SAMPLES):
            episodes = exact_episodes(sample)
            if args.smoke:
                episodes = episodes[:2]
            prepared_all = [prepare_episode(model, episode) for episode in episodes]
            for concept in concepts:
                for prepared in prepared_all:
                    # One permutation per episode. Seeding by position within a
                    # carrier reused the same few draws, and a permutation that
                    # preserves the balanced grouping is a no-op for the reader.
                    rows.append(
                        _episode_row(
                            model, prepared, bank[concept], concept, carrier, seed=len(rows)
                        )
                    )
            print(f"carrier {carrier + 1}/{len(CONFIRM_VISIBLE_SAMPLES)} done", flush=True)

        summary = _summary(rows)
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
        manifest = {
            "schema_version": 1,
            "config": config,
            "config_sha256": _json_sha256(config),
            "raw": out.name,
            "raw_sha256": raw_sha,
            "n_rows": len(rows),
            "elapsed_seconds": time.time() - started,
        }
        for path, value in ((manifest_path, manifest), (summary_path, summary)):
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
    parser.add_argument("--out", type=Path, default=Path("results/matched_reader_v1_raw.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
