"""Run the exact, opaque-codebook causal-neurofeedback experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect import concepts as concept_mod
from introspect import models, retained
from introspect.codebook_icl import (
    CONDITIONS,
    CONFIRM_CONCEPTS,
    CONFIRM_VISIBLE_SAMPLES,
    LABELS,
    VISIBLE_SAMPLES,
    Condition,
    condition_directions,
    exact_episodes,
    prepare_episode,
    score_episode,
    sha256_text,
    tensor_sha256,
)


def _json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return sha256_text(raw)


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _source_files_sha256(root: Path) -> dict[str, str]:
    paths = (
        "src/introspect/codebook_icl.py",
        "src/introspect/hooks.py",
        "src/introspect/concepts.py",
        "src/introspect/models.py",
        "src/introspect/retained.py",
        "scripts/run_codebook_icl.py",
        "pyproject.toml",
        "uv.lock",
    )
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def _git_dirty(root: Path) -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        )
    except Exception:
        return True


def _write_json(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _protocol(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    if args.protocol is None:
        return None, None
    raw = args.protocol.read_text()
    protocol = json.loads(raw)
    for key in (
        "model",
        "model_revision",
        "split",
        "layer",
        "strength",
        "control_seed",
        "max_cosine",
    ):
        setattr(args, key, protocol[key])
    return protocol, sha256_text(raw)


def run(args: argparse.Namespace) -> None:
    protocol, protocol_sha = _protocol(args)
    if args.strength <= 0:
        raise SystemExit("--strength must be positive")

    root = Path(__file__).resolve().parents[1]
    out = args.out
    manifest_path = out.with_suffix(".manifest.json")
    if out.exists() or manifest_path.exists():
        raise SystemExit(f"refusing to overwrite existing result: {out} or {manifest_path}")

    if protocol is not None and args.smoke:
        raise SystemExit("--smoke cannot modify a frozen protocol")
    if protocol is not None:
        concept_names = list(protocol["design"]["concepts"])
        visible_samples = list(protocol["design"]["visible_samples"])
    else:
        concept_names = {
            "dev": list(retained.DEV_CONCEPTS),
            "test": list(retained.TEST_CONCEPTS),
            "confirm": list(CONFIRM_CONCEPTS),
        }[args.split]
        visible_samples = (
            list(CONFIRM_VISIBLE_SAMPLES) if args.split == "confirm" else list(VISIBLE_SAMPLES)
        )
    if args.smoke:
        concept_names = concept_names[:1]
        visible_samples = visible_samples[:1]

    model = models.load(args.model, revision=args.model_revision)
    try:
        if not 0 <= args.layer < model.n_layers:
            raise SystemExit(f"--layer must be between 0 and {model.n_layers - 1}")
        warning = models.memory_warning(args.model)
        if warning:
            print(f"warning: {warning}", file=sys.stderr)

        loaded_revision = models.loaded_revision(model)
        if args.model_revision is not None and loaded_revision != args.model_revision:
            raise SystemExit(
                f"loaded revision {loaded_revision} does not match pinned {args.model_revision}"
            )

        reference = concept_mod.build_bank(
            model, args.layer, list(retained.DEV_CONCEPTS), center=False
        )
        center = torch.stack([direction.vector for direction in reference.values()]).mean(0)
        raw_bank = (
            reference
            if list(concept_names) == list(retained.DEV_CONCEPTS)
            else concept_mod.build_bank(model, args.layer, list(concept_names), center=False)
        )
        bank = {
            name: concept_mod.ConceptVector(
                name=name,
                layer=args.layer,
                vector=direction.vector - center,
            )
            for name, direction in raw_bank.items()
        }
        max_cosine = concept_mod.max_offdiagonal_cosine(bank)
        if max_cosine > args.max_cosine:
            raise SystemExit(
                f"concept vectors are near-collinear: max |cos| {max_cosine:.3f} "
                f"> {args.max_cosine}"
            )

        prepared_by_carrier = []
        for sample in visible_samples:
            episodes = exact_episodes(sample)
            if args.smoke:
                episodes = episodes[:2]
            prepared_by_carrier.append([prepare_episode(model, episode) for episode in episodes])

        if protocol is not None:
            actual_design = {
                "conditions": list(CONDITIONS),
                "concepts": list(concept_names),
                "visible_samples": visible_samples,
                "labels": list(LABELS),
                "centering_concepts": list(retained.DEV_CONCEPTS),
                "n_demos": 4,
                "exact_cells_per_concept_carrier": len(prepared_by_carrier[0]),
                "normalization": "per_position_residual_norm",
            }
            if protocol["design"] != actual_design:
                raise SystemExit("protocol design does not match the executable design")

        source_files = _source_files_sha256(root)
        if protocol is not None and protocol["source_files_sha256"] != source_files:
            raise SystemExit("protocol source hashes do not match the executable source")

        direction_hashes: dict[str, dict[str, str | None]] = {}
        directions_by_concept = {}
        for concept in concept_names:
            directions = condition_directions(bank[concept], control_seed=args.control_seed)
            directions_by_concept[concept] = directions
            direction_hashes[concept] = {
                condition: tensor_sha256(direction.vector) if direction is not None else None
                for condition, direction in directions.items()
            }

        config: dict[str, object] = {
            "schema_version": 3,
            "estimand": "opaque_label_accuracy_from_causally_varied_hidden_state",
            "model_requested": args.model,
            "model_resolved": model.name,
            "model_revision": loaded_revision,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "split": args.split,
            "concepts": list(concept_names),
            "layer": args.layer,
            "strength": args.strength,
            "control_seed": args.control_seed,
            "conditions": list(CONDITIONS),
            "labels": list(LABELS),
            "visible_samples": visible_samples,
            "exact_cells_per_concept_carrier": len(prepared_by_carrier[0]),
            "direction_sha256": direction_hashes,
            "centering_direction_sha256": tensor_sha256(center),
            "centering_concepts": list(retained.DEV_CONCEPTS),
            "normalization": "per_position_residual_norm",
            "max_offdiagonal_cosine": max_cosine,
            "source_files_sha256": source_files,
            "source_sha256": _json_sha256(source_files),
            "prompt_set_sha256": _json_sha256(
                sorted(
                    {
                        prepared.prompt_sha256
                        for carrier in prepared_by_carrier
                        for prepared in carrier
                    }
                )
            ),
            "protocol": protocol,
            "protocol_sha256": protocol_sha,
            "smoke": args.smoke,
            "git_commit": _git_commit(root),
            "git_dirty": _git_dirty(root),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        }
        config_sha = _json_sha256(config)
        correct: dict[Condition, int] = {condition: 0 for condition in CONDITIONS}
        rows = 0
        started = time.time()

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp")
        with tmp.open("w") as handle:
            for concept_id, concept in enumerate(concept_names):
                directions = directions_by_concept[concept]
                for carrier_id, prepared_episodes in enumerate(prepared_by_carrier):
                    for prepared in prepared_episodes:
                        scores: dict[str, object] = {}
                        for condition in CONDITIONS:
                            score = score_episode(
                                model,
                                prepared,
                                condition,
                                directions[condition],
                                strength=args.strength,
                            )
                            correct[condition] += int(score.correct)
                            scores[condition] = {
                                "direction_sha256": direction_hashes[concept][condition],
                                "predicted_label": score.predicted_label,
                                "correct": score.correct,
                                "conditional_probs": score.conditional_probs,
                                "full_logprobs": score.full_logprobs,
                                "label_mass": score.label_mass,
                                "format_ok": score.format_ok,
                            }
                        token_ids = prepared.input_ids[0].tolist()
                        row = {
                            "schema_version": 3,
                            "config_sha256": config_sha,
                            "concept": concept,
                            "carrier_id": carrier_id,
                            "cell_id": prepared.episode.cell_id,
                            "episode_sha256": prepared.episode.digest(),
                            "prompt": prepared.prompt,
                            "prompt_sha256": prepared.prompt_sha256,
                            "token_ids_sha256": _json_sha256(token_ids),
                            "token_ids": token_ids,
                            "state_token_positions": prepared.state_positions,
                            "demo_signs": prepared.episode.demo_signs,
                            "query_sign": prepared.episode.query_sign,
                            "label_mapping": {
                                "+1": prepared.episode.positive_label,
                                "-1": prepared.episode.negative_label,
                            },
                            "correct_label": prepared.episode.correct_label,
                            "condition_scores": scores,
                        }
                        handle.write(json.dumps(row, sort_keys=True) + "\n")
                        rows += 1
                done = (concept_id + 1) * len(visible_samples) * len(prepared_by_carrier[0])
                print(
                    f"concept {concept_id + 1}/{len(concept_names)} {concept}: "
                    + " ".join(f"{c}={correct[c]}/{done}" for c in CONDITIONS)
                    + f" [{time.time() - started:.0f}s]",
                    flush=True,
                )
        tmp.replace(out)

        raw_sha = hashlib.sha256(out.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 3,
            "config": config,
            "config_sha256": config_sha,
            "raw": out.name,
            "raw_sha256": raw_sha,
            "n_episode_rows": rows,
            "n_scored_forwards": rows * len(CONDITIONS),
            "correct_by_condition": correct,
        }
        _write_json(manifest_path, manifest)
        print(f"wrote {out} (sha256={raw_sha})")
        print(f"manifest {manifest_path}")
    finally:
        model.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-0.5b")
    parser.add_argument("--model-revision")
    parser.add_argument("--split", choices=("dev", "test", "confirm"), default="dev")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--control-seed", type=int, default=0)
    parser.add_argument("--max-cosine", type=float, default=0.5)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("results/codebook_icl_dev.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
