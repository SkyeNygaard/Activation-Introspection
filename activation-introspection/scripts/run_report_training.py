"""Train a zero-demonstration activation reporter and score it on held-out banks.

The frozen causal-codebook result showed that four in-context demonstrations can
teach an episode-specific mapping from a causally injected hidden state to an
opaque label. This runner asks whether *training* can remove the demonstrations:
a LoRA is fitted on one bank of concept directions under a single fixed mapping,
then scored on a disjoint bank of directions and on carriers it never saw.

The design's load-bearing property is structural. Both members of every query
twin are byte-identical in visible text and have opposite correct labels, so any
prompt-only strategy scores exactly 0.500 on twin pairs whatever the model does.
The measurement is therefore twin-pair accuracy, not per-row accuracy.

Everything is frozen before the model is loaded: the protocol is written on first
run and verified by hash afterwards, the raw artifact is written atomically and
refuses to overwrite, and the manifest pins the model revision, source hashes,
directions, prompts, and row count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect import models
from introspect.concepts import (
    ConceptVector,
    build_bank,
    build_concept_vector,
    max_offdiagonal_cosine,
    random_control,
    shuffled_control,
)
from introspect.hooks import intervene
from introspect.ift import attach_lora
from introspect.report_training import (
    CENTERING_CONCEPTS,
    CONDITIONS,
    EVAL_CARRIERS,
    EVAL_CONCEPTS,
    LABELS,
    TRAIN_CARRIERS,
    TRAIN_CONCEPTS,
    PreparedReport,
    json_sha256,
    label_for,
    prepare_report,
    score_logits,
    sha256_text,
    sign_intervention,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LAYER = 9
STRENGTH = 1.0
CONTROL_SEED = 0
MAX_COSINE = 0.5
EPOCHS = 6
LEARNING_RATE = 1e-4
LORA_RANK = 16
LORA_ALPHA = 32
TRAIN_SEED = 0

_SOURCE_PATHS = (
    "scripts/run_report_training.py",
    "src/introspect/report_training.py",
    "src/introspect/concepts.py",
    "src/introspect/hooks.py",
    "src/introspect/ift.py",
    "src/introspect/models.py",
    "pyproject.toml",
    "uv.lock",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in _SOURCE_PATHS}


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    shape = json.dumps(list(value.shape), separators=(",", ":")).encode()
    return hashlib.sha256(shape + value.numpy().tobytes()).hexdigest()


def build_protocol(smoke: bool) -> dict[str, Any]:
    return {
        "analysis_rules": {
            "aggregation": (
                "score twin pairs, not rows: a pair is correct only when both "
                "byte-identical members get their opposite labels right; "
                "aggregate as an unweighted mean over concept-carrier strata"
            ),
            "clean_role": (
                "the no-injection arm has no correct label; it measures standing "
                "label bias and is not scored for accuracy"
            ),
            "inference_unit": (
                "concept crossed with carrier; the two signs of a twin are one "
                "paired observation, not two independent samples"
            ),
            "input_only_identity": (
                "twin members are byte-identical with opposite correct labels, so "
                "any prompt-only strategy scores exactly 0.500 on pairs by "
                "construction; no separate input-only arm is run because the "
                "value is an identity rather than a measurement"
            ),
            "primary_gate": (
                "trained target twin-pair accuracy on held-out concepts exceeds "
                "0.50, the untrained base model does not, and the trained "
                "target arm exceeds the stronger of the random and shuffled arms"
            ),
            "verbalization_gate": (
                "the trained arm must retain at least 0.90 full-vocabulary "
                "format rate and at least 0.50 mean label mass; a discriminative "
                "readout over tokens the model would not emit is not a "
                "verbalization, and V1 failed exactly this way"
            ),
            "seed_inference": (
                "the independent unit for the trained arm is the training seed; "
                "report every planned seed and their range, and never quote a "
                "single seed as the effect"
            ),
            "no_retuning": (
                "do not change model, layer, strength, prompts, labels, banks, "
                "carriers, optimizer settings or gates after inspecting results; "
                "correct only invalidating bugs, disclose them, and rerun"
            ),
        },
        "claim_boundary": (
            "A positive result shows that LoRA training produces a report that "
            "tracks the sign of a causally injected residual edit on directions "
            "and carriers withheld from training, with visible text held "
            "constant. It does not establish privileged self-access, "
            "verbalization of naturally occurring computation, faithfulness "
            "under adversarial pressure, or transfer to other layers, models, "
            "strengths, or non-binary internal variables."
        ),
        "design": {
            "centering_concepts": list(CENTERING_CONCEPTS),
            "conditions": list(CONDITIONS),
            "control_seed": CONTROL_SEED,
            "device": "mps",
            "dtype_base": "bfloat16",
            "dtype_adapter": "float32",
            "epochs": EPOCHS,
            "eval_carriers": list(EVAL_CARRIERS),
            "eval_concepts": list(EVAL_CONCEPTS),
            "injection_layer": LAYER,
            "labels": list(LABELS),
            "learning_rate": LEARNING_RATE,
            "lora_alpha": LORA_ALPHA,
            "lora_rank": LORA_RANK,
            "loss": "full_vocabulary_cross_entropy_on_the_label_token",
            "mapping": {"+1": label_for(1), "-1": label_for(-1)},
            "max_offdiagonal_cosine_gate": MAX_COSINE,
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "smoke": smoke,
            "strength": STRENGTH,
            "train_carriers": list(TRAIN_CARRIERS),
            "train_concepts": list(TRAIN_CONCEPTS),
            "train_seeds_planned": [0, 1, 2, 3],
        },
        "frozen_on": "2026-08-10",
        "source_files_sha256": _source_hashes(),
        "stop": (
            "report the null if the trained target arm does not clear 0.50 on "
            "held-out concepts; report generic perturbation-sign readout rather "
            "than concept-specific reporting if the random and shuffled arms "
            "match the target arm"
        ),
    }


def freeze_protocol(path: Path, smoke: bool) -> tuple[dict[str, Any], str]:
    protocol = build_protocol(smoke)
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != protocol:
            raise SystemExit(
                f"{path} differs from the protocol this source would freeze; "
                "issue a new version rather than editing a frozen one"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol, _sha256(path)


def _centered_bank(
    model: models.LoadedModel, concepts: tuple[str, ...], center: torch.Tensor
) -> dict[str, ConceptVector]:
    """Build directions and subtract a center estimated on a third bank.

    Estimating the center inside the evaluation bank would make each test
    direction depend on the others; the frozen V2 protocol found and fixed that,
    and the same rule applies here.
    """
    bank = {name: build_concept_vector(model, name, LAYER) for name in concepts}
    return {
        name: ConceptVector(name=cv.name, layer=cv.layer, vector=cv.vector - center)
        for name, cv in bank.items()
    }


def train_reporter(
    model: models.LoadedModel,
    bank: dict[str, ConceptVector],
    prepared: dict[str, PreparedReport],
    seed: int,
) -> list[float]:
    """Fit the adapter on signed edits over the training bank and carriers."""
    import random

    examples = [
        (concept, carrier, sign)
        for concept in sorted(bank)
        for carrier in TRAIN_CARRIERS
        for sign in (-1, 1)
    ]
    rng = random.Random(seed)
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    losses: list[float] = []
    model.model.train()
    try:
        for epoch in range(EPOCHS):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for index in order:
                concept, carrier, sign = examples[index]
                report = prepared[carrier]
                edits = sign_intervention(
                    bank[concept].vector,
                    LAYER,
                    report.marker_position,
                    sign,
                    strength=STRENGTH,
                    label=f"train:{concept}:{sign:+d}",
                )
                with intervene(model, edits, prompt_len=int(report.input_ids.shape[1])):
                    logits = cast(Any, model.model)(report.input_ids).logits
                # Full-vocabulary cross-entropy, not a two-way softmax over the
                # label logits. V1 restricted the loss to the two options, which
                # left the rest of the vocabulary unconstrained: the adapter
                # ordered Q against K correctly while driving the labels' own
                # probability mass to ~5e-9, so the "report" was a forced choice
                # among tokens the model would never emit. See notes/07.
                token_id = report.label_ids[0 if sign == 1 else 1]
                loss = torch.nn.functional.cross_entropy(
                    logits[0, -1].float().unsqueeze(0),
                    torch.tensor([token_id], device=model.device),
                )
                cast(Any, loss).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
            window = losses[-len(examples) :]
            print(
                f"  epoch {epoch + 1}/{EPOCHS}  loss {sum(window) / len(window):.4f}",
                flush=True,
            )
    finally:
        model.model.eval()
    return losses


@torch.no_grad()
def score_all(
    model: models.LoadedModel,
    bank: dict[str, ConceptVector],
    prepared: dict[str, PreparedReport],
    *,
    trained: bool,
    carriers: tuple[str, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in sorted(bank):
        target = bank[concept]
        directions: dict[str, ConceptVector | None] = {
            "target": target,
            "random": random_control(target, seed=CONTROL_SEED),
            "shuffled": shuffled_control(target, seed=CONTROL_SEED),
            "clean": None,
        }
        for carrier in carriers:
            report = prepared[carrier]
            for condition in CONDITIONS:
                direction = directions[condition]
                signs = (0,) if condition == "clean" else (-1, 1)
                for sign in signs:
                    edits = (
                        []
                        if direction is None
                        else sign_intervention(
                            direction.vector,
                            LAYER,
                            report.marker_position,
                            sign,
                            strength=STRENGTH,
                            label=f"{condition}:{concept}:{sign:+d}",
                        )
                    )
                    with intervene(model, edits, prompt_len=int(report.input_ids.shape[1])):
                        logits = cast(Any, model.model)(report.input_ids).logits
                    score = score_logits(logits, report, sign)
                    rows.append(
                        {
                            "arm": "trained" if trained else "base",
                            "condition": condition,
                            "concept": concept,
                            "carrier": carrier,
                            "carrier_sha256": sha256_text(carrier),
                            "prompt_sha256": report.prompt_sha256,
                            "marker_position": report.marker_position,
                            "sign": sign,
                            "expected_label": None if sign == 0 else label_for(sign),
                            "direction_sha256": (
                                None if direction is None else _tensor_sha256(direction.vector)
                            ),
                            "predicted_label": score.predicted_label,
                            "correct": score.correct,
                            "correct_probability": score.correct_probability,
                            "signed_margin": score.signed_margin,
                            "label_mass": score.label_mass,
                            "format_ok": score.format_ok,
                        }
                    )
    return rows


def _write_atomic(path: Path, lines: list[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    created = False
    try:
        with temporary.open("x") as handle:
            created = True
            for line in lines:
                handle.write(line + "\n")
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "results" / "report_training_protocol_v3.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; reproduce into a new path")

    _protocol, protocol_sha = freeze_protocol(args.protocol, args.smoke)
    source_hashes = _source_hashes()
    train_concepts = TRAIN_CONCEPTS[:2] if args.smoke else TRAIN_CONCEPTS
    eval_concepts = EVAL_CONCEPTS[:2] if args.smoke else EVAL_CONCEPTS
    eval_carriers = EVAL_CARRIERS[:1] if args.smoke else EVAL_CARRIERS

    started = time.time()
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    resolved = models.loaded_revision(model)
    if resolved not in {MODEL_REVISION, "unknown"}:
        raise SystemExit(f"expected revision {MODEL_REVISION}, loaded {resolved}")
    model.model.to(torch.bfloat16)
    object.__setattr__(model, "dtype", torch.bfloat16)

    centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
    center = torch.stack([cv.vector for cv in centering.values()]).mean(dim=0)
    train_bank = _centered_bank(model, train_concepts, center)
    eval_bank = _centered_bank(model, eval_concepts, center)
    for name, bank in (("train", train_bank), ("eval", eval_bank)):
        worst = max_offdiagonal_cosine(bank)
        if worst > MAX_COSINE:
            raise SystemExit(f"{name} bank is degenerate: max |cos| {worst:.3f}")
        print(f"{name} bank max |cos| {worst:.3f}", flush=True)

    carriers = tuple(dict.fromkeys((*TRAIN_CARRIERS, *eval_carriers)))
    prepared = {carrier: prepare_report(model, carrier) for carrier in carriers}

    print("scoring untrained base model", flush=True)
    base_rows = score_all(model, eval_bank, prepared, trained=False, carriers=eval_carriers)

    print(f"attaching adapter and training (seed {args.seed})", flush=True)
    # Seed the adapter initialisation as well as the example order: the point of
    # running several seeds is to vary exactly what a single run cannot.
    torch.manual_seed(args.seed)
    attach_lora(model, r=LORA_RANK, alpha=LORA_ALPHA)
    for parameter in model.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    losses = train_reporter(model, train_bank, prepared, args.seed)

    print("scoring trained model on held-out banks", flush=True)
    trained_rows = score_all(model, eval_bank, prepared, trained=True, carriers=eval_carriers)
    print("scoring trained model on the training bank", flush=True)
    seen_rows = [
        row | {"arm": "trained_seen_bank"}
        for row in score_all(model, train_bank, prepared, trained=True, carriers=eval_carriers)
    ]

    if source_hashes != _source_hashes():
        raise SystemExit("source files changed during the run; artifact discarded")

    rows = base_rows + trained_rows + seen_rows
    lines = [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows]
    _write_atomic(args.out, lines)
    raw_sha = _sha256(args.out)

    config = {
        "centering_direction_sha256": _tensor_sha256(center),
        "device": "mps",
        "dtype_base": "bfloat16",
        "elapsed_seconds": round(time.time() - started, 1),
        "eval_carriers": list(eval_carriers),
        "eval_concepts": list(eval_concepts),
        "final_epoch_mean_loss": sum(losses[-len(train_bank) * len(TRAIN_CARRIERS) * 2 :])
        / max(len(train_bank) * len(TRAIN_CARRIERS) * 2, 1),
        "git_commit": _git("rev-parse", "HEAD"),
        "train_seed": args.seed,
        "git_dirty": bool(_git("status", "--porcelain")),
        "model_resolved": model.name,
        "model_revision": resolved,
        "n_train_steps": len(losses),
        "platform": platform.platform(),
        "prompt_set_sha256": json_sha256(
            {carrier: prepared[carrier].prompt_sha256 for carrier in eval_carriers}
        ),
        "protocol_sha256": protocol_sha,
        "python": platform.python_version(),
        "smoke": args.smoke,
        "source_files_sha256": source_hashes,
        "source_sha256": json_sha256(source_hashes),
        "torch": torch.__version__,
        "train_carriers": list(TRAIN_CARRIERS),
        "train_concepts": list(train_concepts),
    }
    manifest = {
        "config": config,
        "config_sha256": json_sha256(config),
        "n_rows": len(rows),
        "raw": args.out.name,
        "raw_sha256": raw_sha,
        "schema_version": 1,
        "loss_curve": losses,
    }
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {args.out} ({len(rows)} rows, sha256 {raw_sha[:12]}…) "
        f"in {config['elapsed_seconds']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
