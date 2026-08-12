"""Does the cheap reader only win because it reads at the injection site?

This attacks [`notes/11`](../notes/11-matched-cost-reader.md) rather than
extending it. That result found a four-shot nearest-centroid reader at 1.000
against the model's 0.892 on the same 576 episodes, with zero episodes where the
model wins. Its stated first limitation is that the reader reads the residual at
block 9 — the block whose output the injection edits — where the planted signal is
maximal by construction.

So read everywhere. One forward pass per episode, the same injection at block 9,
and the same reader fitted separately on the states captured after every block.
Blocks below the injection site are a free validity control: nothing has been
edited yet, so a reader there must be at chance.

The pre-registered fork is stated in the protocol. If the reader still beats the
model at the output, the dissociation is real and the site is not doing the work.
If the reader decays to the model's level, then the model's performance tracks
the signal actually available late in the stack, and `notes/11`'s dominance is an
artifact of where it read.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_matched_reader import (
    FROZEN_TARGET_ACCURACY,
    LAYER,
    MODEL,
    MODEL_REVISION,
    REPRODUCTION_TOLERANCE,
    STRENGTH,
    _git,
    _json_sha256,
    _read,
    _sha256,
)

from introspect import concepts as concept_mod
from introspect import models, retained
from introspect.codebook_icl import (
    CONFIRM_CONCEPTS,
    CONFIRM_VISIBLE_SAMPLES,
    LABELS,
    condition_interventions,
    exact_episodes,
    prepare_episode,
)
from introspect.hooks import capture, intervene
from introspect.preflight import check as preflight_check

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/run_reader_depth.py",
    "scripts/run_matched_reader.py",
    "src/introspect/codebook_icl.py",
    "src/introspect/concepts.py",
    "src/introspect/hooks.py",
    "src/introspect/models.py",
    "pyproject.toml",
    "uv.lock",
)


def _protocol(n_layers: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "frozen_on": "2026-08-11",
        "role": (
            "adversarial falsification of notes/11, not an extension of it. The "
            "hypothesis under attack is this repository's own new result."
        ),
        "question": (
            "Is the four-shot reader's dominance over the model an artifact of reading "
            "the residual at the block the injection edits?"
        ),
        "design": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "injection_layer": LAYER,
            "strength": STRENGTH,
            "read_layers": list(range(n_layers)),
            "concepts": list(CONFIRM_CONCEPTS),
            "visible_samples": list(CONFIRM_VISIBLE_SAMPLES),
            "labels": list(LABELS),
            "unit": "episode; the same 576 episodes as the frozen confirmation",
            "reader": "nearest centroid, Euclidean, four demonstrations",
        },
        "validity_gate": {
            "pre_injection_reader_at_chance": 0.60,
            "rule": (
                "blocks strictly below the injection site carry no edit, so the reader "
                "there must not exceed 0.60. If it does, the reader is reading something "
                "the intervention did not put there and every number here is void."
            ),
        },
        "reproduction_gate": {
            "frozen_target_accuracy": FROZEN_TARGET_ACCURACY,
            "tolerance": REPRODUCTION_TOLERANCE,
        },
        "prereg_fork": {
            "reader_beats_model_at_output_layer": (
                "notes/11 survives its own stated limitation. The signal remains "
                "linearly available at the last block while the model's use of it does "
                "not improve, which localizes the storage/use dissociation instead of "
                "asserting it."
            ),
            "reader_decays_to_model_level": (
                "notes/11's dominance is an artifact of the read site. The model's "
                "accuracy then tracks the signal actually available downstream, the "
                "privileged-access verdict weakens to 'not established at the injection "
                "site', and that limitation becomes the finding."
            ),
            "primary_statistic": (
                "reader accuracy at the final block, and the deepest block at which the "
                "reader still exceeds the model"
            ),
        },
        "source_files_sha256": {path: _sha256(ROOT / path) for path in SOURCE_PATHS},
    }


def _freeze_protocol(path: Path, n_layers: int) -> tuple[dict[str, object], str]:
    protocol = _protocol(n_layers)
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise SystemExit(f"{path} differs from this source; issue a new protocol version")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol, _sha256(path)


def _summary(rows: list[dict[str, object]], n_layers: int) -> dict[str, object]:
    total = len(rows)
    model_accuracy = sum(bool(row["model_correct"]) for row in rows) / total
    by_layer = {
        str(layer): sum(bool(row["reader_correct_by_layer"][layer]) for row in rows) / total  # type: ignore[index]
        for layer in range(n_layers)
    }
    pre = [by_layer[str(layer)] for layer in range(LAYER)]
    deepest = max(
        (layer for layer in range(n_layers) if by_layer[str(layer)] > model_accuracy),
        default=None,
    )
    final = by_layer[str(n_layers - 1)]
    valid = max(pre) <= 0.60
    reproduced = abs(model_accuracy - FROZEN_TARGET_ACCURACY) <= REPRODUCTION_TOLERANCE
    return {
        "n_episodes": total,
        "model_target_accuracy": model_accuracy,
        "reader_accuracy_by_read_layer": by_layer,
        "max_pre_injection_reader_accuracy": max(pre),
        "reader_accuracy_at_injection_layer": by_layer[str(LAYER)],
        "reader_accuracy_at_final_layer": final,
        "deepest_layer_reader_beats_model": deepest,
        "gates": {
            "pre_injection_reader_at_chance": valid,
            "reproduces_frozen_confirmation": reproduced,
        },
        "verdict": (
            "void_validity_gate_failed"
            if not valid
            else "void_did_not_reproduce"
            if not reproduced
            else "dissociation_survives_at_output"
            if final > model_accuracy
            else "dominance_is_a_read_site_artifact"
        ),
    }


def run(args: argparse.Namespace) -> None:
    out = args.out
    manifest_path = out.with_suffix(".manifest.json")
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    for path in (out, manifest_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    preflight_check(MODEL, training=False)
    model = models.load(MODEL, revision=MODEL_REVISION)
    started = time.time()
    rows: list[dict[str, object]] = []
    try:
        if models.loaded_revision(model) != MODEL_REVISION:
            raise SystemExit("loaded model revision does not match the frozen revision")
        n_layers = model.n_layers
        protocol, protocol_sha = _freeze_protocol(
            args.protocol or Path("results/reader_depth_protocol_v1.json"), n_layers
        )
        read_layers = list(range(n_layers))

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
                    episode = prepared.episode
                    interventions = condition_interventions(
                        "target",
                        bank[concept],
                        prepared.state_positions,
                        episode.state_signs,
                        strength=STRENGTH,
                    )
                    with (
                        intervene(
                            model, interventions, prompt_len=int(prepared.input_ids.shape[1])
                        ),
                        capture(model, read_layers) as store,
                    ):
                        logits = model.forward_logits(prepared.input_ids)[0, -1].float().cpu()
                    predicted = LABELS[int(logits[torch.tensor(prepared.label_ids)].argmax())]
                    correct_by_layer = []
                    for layer in read_layers:
                        states = store.acts[layer][0][0, list(prepared.state_positions)]
                        sign = _read(states, episode.state_signs, "centroid_euclidean", 0)
                        correct_by_layer.append(episode.label_for(sign) == episode.correct_label)
                    rows.append(
                        {
                            "concept": concept,
                            "carrier": carrier,
                            "cell_id": episode.cell_id,
                            "correct_label": episode.correct_label,
                            "prompt_sha256": prepared.prompt_sha256,
                            "model_predicted": predicted,
                            "model_correct": predicted == episode.correct_label,
                            "reader_correct_by_layer": correct_by_layer,
                        }
                    )
            print(f"carrier {carrier + 1}/{len(CONFIRM_VISIBLE_SAMPLES)} done", flush=True)

        summary = _summary(rows, n_layers)
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
            "injection_layer": LAYER,
            "n_layers": n_layers,
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
    parser.add_argument("--out", type=Path, default=Path("results/reader_depth_v1_raw.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
