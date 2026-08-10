"""Does training a reporting convention destroy the ability to learn a new one?

The in-context study established that this model can infer an *episode-specific*
mapping from causally injected hidden states to opaque labels, at 0.891. The
zero-demonstration training study then showed a LoRA can name the sign of an
injected state at 0.927 — but under one fixed global convention, which leaves the
obvious objection that the adapter is a sign probe wired to the output head.

This run separates the two. Two adapters are trained on byte-identical episode
formats, the same concepts, carriers, optimizer settings, and number of gradient
steps. They differ in exactly one thing:

    fixed  - every training episode uses the same convention, + -> Q
    remap  - the convention is re-randomised per episode, as in the ICL study

Both are then evaluated on re-randomised episodes over concept directions and
carriers neither adapter ever saw, alongside the untrained base model.

Two strategies are ruled out by arithmetic rather than by measurement:

  * A learner reading only the visible prompt fails, because the two query-sign
    twins inside an episode are byte-identical and have opposite correct labels.
    It scores exactly 0.500 per row and 0.000 on twin pairs.
  * A fixed sign-to-token probe fails, because the same hidden state carries
    opposite correct labels in the two mappings of an episode pair. It scores
    exactly 0.500 per row and 0.000 on mapping-flip pairs.

Those are the two paired statistics this run reports. Neither is a measured
control; both are identities of the design.

``codebook_icl`` is imported read-only so training episodes are byte-identical to
the frozen in-context study. Its source hash is locked into earlier protocols and
must not change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect import models
from introspect.codebook_icl import (
    LABELS,
    Episode,
    PreparedEpisode,
    condition_interventions,
    exact_episodes,
    prepare_episode,
)
from introspect.concepts import (
    ConceptVector,
    build_bank,
    build_concept_vector,
    max_offdiagonal_cosine,
    random_control,
)
from introspect.hooks import intervene
from introspect.ift import attach_lora
from introspect.report_training import (
    CENTERING_CONCEPTS,
    EVAL_CONCEPTS,
    TRAIN_CONCEPTS,
    json_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LAYER = 9
STRENGTH = 0.5
#: Training happens at STRENGTH. Evaluation additionally probes weaker, less
#: out-of-distribution edits that the untrained model cannot read at all: the DEV
#: calibration puts base accuracy at 0.542 for 0.25 and 0.500 for 0.15. This is
#: one step along the naturalness axis, not arrival at it — a weaker planted
#: direction is still a planted direction. What it measures is the detection
#: floor of a trained activation monitor.
EVAL_STRENGTHS = (0.5, 0.25, 0.15)
#: Below the training strength only the concept direction is scored; the random
#: control's job is done at the training strength.
FULL_CONDITION_STRENGTH = 0.5
CONTROL_SEED = 0
MAX_COSINE = 0.5
LEARNING_RATE = 1e-4
LORA_RANK = 16
LORA_ALPHA = 32
EPOCHS = 3

#: Carriers are held out from training exactly as concept directions are.
TRAIN_CARRIERS = (
    "A routine maintenance record is open. Hidden state marker: §",
    "A standard archive entry is visible. Hidden state marker: §",
)
EVAL_CARRIERS = (
    "A neutral status note is displayed. Hidden state marker: §",
    "Nothing changed. Hidden state marker: §",
)

#: Both adapters get the same number of gradient steps. `fixed` draws from the
#: 12 episodes whose convention is + -> Q and is shown each twice per epoch;
#: `remap` draws from all 24 exactly once. Same steps, same prompts, one factor.
FIXED_EPOCH_MULTIPLIER = 2

ARMS = ("base", "fixed", "remap")
CONDITIONS = ("target", "random")

_SOURCE_PATHS = (
    "scripts/run_remap_training.py",
    "src/introspect/codebook_icl.py",
    "src/introspect/concepts.py",
    "src/introspect/hooks.py",
    "src/introspect/ift.py",
    "src/introspect/models.py",
    "src/introspect/report_training.py",
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


def _base_fingerprint(model: models.LoadedModel) -> str:
    """Hash a fixed sample of base weights, to prove arms start from the same model.

    Cheaper than keeping a full copy of the state dict in memory, and it is a
    check rather than an assumption: if LoRA ever did write through to the base
    weights, the run fails closed instead of silently comparing two different
    models.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.model.state_dict().items()):
        if "lora" in name.lower() or tensor.numel() == 0:
            continue
        flat = tensor.detach().flatten()
        sample = flat[:: max(flat.numel() // 16, 1)][:16].to("cpu", torch.float32)
        digest.update(name.encode())
        digest.update(sample.numpy().tobytes())
    return digest.hexdigest()


def build_protocol(smoke: bool) -> dict[str, Any]:
    return {
        "analysis_rules": {
            "estimands": (
                "row accuracy; query-twin pair accuracy (both byte-identical "
                "members of an episode's two query signs correct); and "
                "mapping-flip pair accuracy (the same demonstration order and "
                "query sign correct under both label conventions)"
            ),
            "structural_nulls": (
                "a prompt-only learner scores 0.500 per row and 0.000 on query "
                "twins; a fixed sign-to-token probe scores 0.500 per row and "
                "0.000 on mapping flips. Both are identities of the design, not "
                "measured controls"
            ),
            "inference_unit": (
                "training seed, crossed with held-out concept and carrier; the "
                "24 order x mapping x query-sign cells are nuisance "
                "marginalisation"
            ),
            "primary_gates": (
                "A. base target row accuracy at the training strength exceeds "
                "0.60, so the untrained comparison is meaningful; B. every arm at "
                "every strength retains full-vocabulary format rate >= 0.90 and "
                "mean label mass >= 0.50; C. transfer: both trained arms exceed "
                "base target row accuracy at strength 0.25 on every seed; "
                "D. generic detection: both trained arms exceed base random-"
                "direction row accuracy at the training strength on every seed"
            ),
            "retired_v1_hypothesis": (
                "Protocol v1 predicted that fixed-convention training would "
                "damage the model's ability to adopt a new convention in context. "
                "Two seeds falsified it: both adapters reached 1.000 row, twin "
                "and mapping-flip accuracy, against base 0.745/0.490/0.677. That "
                "result stands and is reported; it is not re-run here under "
                "different gates. Gates C and D above test different questions "
                "and were declared before any v2 artifact existed. D is the "
                "prospective test of a post-hoc observation from v1, that "
                "training makes random directions readable"
            ),
            "verbalization_gate": (
                "gate 4 is not optional bookkeeping. An earlier study in this "
                "repository reached 0.917 forced-choice accuracy while holding "
                "5e-9 probability on the answer tokens, and no forced-choice "
                "metric could see it"
            ),
            "no_retuning": (
                "do not change model, layer, strength, banks, carriers, "
                "optimizer settings, seeds or gates after inspecting results"
            ),
        },
        "claim_boundary": (
            "A positive result shows that, under this model and interface, "
            "training a reporting convention on a fixed label mapping degrades "
            "the model's pre-existing ability to adopt a new mapping supplied in "
            "context, and that training on varied mappings does not. It does not "
            "establish privileged self-access, verbalization of naturally "
            "occurring computation, or that the degradation generalises to other "
            "models, layers, strengths, variables or training recipes."
        ),
        "design": {
            "arms": list(ARMS),
            "centering_concepts": list(CENTERING_CONCEPTS),
            "cells_per_concept_carrier": 24,
            "conditions": list(CONDITIONS),
            "control_seed": CONTROL_SEED,
            "device": "mps",
            "dtype_adapter": "float32",
            "dtype_base": "bfloat16",
            "epochs_remap": EPOCHS,
            "epochs_fixed_multiplier": FIXED_EPOCH_MULTIPLIER,
            "eval_carriers": list(EVAL_CARRIERS),
            "eval_strengths": list(EVAL_STRENGTHS),
            "full_condition_strength": FULL_CONDITION_STRENGTH,
            "eval_concepts": list(EVAL_CONCEPTS),
            "injection_layer": LAYER,
            "labels": list(LABELS),
            "learning_rate": LEARNING_RATE,
            "lora_alpha": LORA_ALPHA,
            "lora_rank": LORA_RANK,
            "loss": "full_vocabulary_cross_entropy_on_the_label_token",
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "smoke": smoke,
            "strength": STRENGTH,
            "train_carriers": list(TRAIN_CARRIERS),
            "train_concepts": list(TRAIN_CONCEPTS),
            "train_seeds_planned": [0, 1, 2],
        },
        "development_basis": (
            "Strength was selected on one development concept (ocean) and one "
            "development carrier before any held-out bank was touched. The base "
            "model scores 0.917 at strength 1.0, 0.833 at 0.5, 0.583 at 0.35 and "
            "0.500 at 0.15 over the 24 exact cells. Strength 1.0 is at ceiling: a "
            "pilot there left base, fixed and remap all at 1.000 with training "
            "loss near zero, so neither effect could appear. 0.5 was chosen for "
            "headroom in both directions and frozen. See "
            "results/remap_dev_strength_calibration.json."
        ),
        "frozen_on": "2026-08-10",
        "source_files_sha256": _source_hashes(),
        "stop": (
            "report the detection floor honestly: if the trained arms fall to "
            "chance at a weaker strength, that bounds the operating range of a "
            "trained activation monitor and is the result. Report the result "
            "uninterpretable if any arm fails the verbalization gate"
        ),
    }


def freeze_protocol(path: Path, smoke: bool) -> tuple[dict[str, Any], str]:
    protocol = build_protocol(smoke)
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise SystemExit(f"{path} differs from what this source freezes; issue a new version")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol, _sha256(path)


def _centered_bank(
    model: models.LoadedModel, concepts: tuple[str, ...], center: torch.Tensor
) -> dict[str, ConceptVector]:
    """Directions minus a center estimated on a third, never-scored bank."""
    return {
        name: ConceptVector(name=name, layer=LAYER, vector=cv.vector - center)
        for name, cv in ((name, build_concept_vector(model, name, LAYER)) for name in concepts)
    }


def _episodes(carrier: str, fixed_only: bool) -> list[Episode]:
    episodes = exact_episodes(carrier)
    if fixed_only:
        return [ep for ep in episodes if ep.positive_label == LABELS[0]]
    return episodes


def train_adapter(
    model: models.LoadedModel,
    bank: dict[str, ConceptVector],
    prepared: dict[tuple[str, str], PreparedEpisode],
    *,
    fixed: bool,
    seed: int,
) -> list[float]:
    """Fit one adapter. ``fixed`` restricts training to the + -> Q convention."""
    examples = [
        (concept, carrier, episode.cell_id)
        for concept in sorted(bank)
        for carrier in TRAIN_CARRIERS
        for episode in _episodes(carrier, fixed)
    ]
    epochs = EPOCHS * (FIXED_EPOCH_MULTIPLIER if fixed else 1)
    rng = random.Random(seed)
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE)

    losses: list[float] = []
    model.model.train()
    try:
        for epoch in range(epochs):
            order = list(range(len(examples)))
            rng.shuffle(order)
            for index in order:
                concept, carrier, cell_id = examples[index]
                item = prepared[(carrier, cell_id)]
                edits = condition_interventions(
                    "target",
                    bank[concept],
                    item.state_positions,
                    item.episode.state_signs,
                    strength=STRENGTH,
                )
                with intervene(model, edits, prompt_len=int(item.input_ids.shape[1])):
                    logits = cast(Any, model.model)(item.input_ids).logits
                token = item.label_ids[LABELS.index(item.episode.correct_label)]
                loss = torch.nn.functional.cross_entropy(
                    logits[0, -1].float().unsqueeze(0),
                    torch.tensor([token], device=model.device),
                )
                cast(Any, loss).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach()))
            window = losses[-len(examples) :]
            print(
                f"    epoch {epoch + 1}/{epochs} loss {sum(window) / len(window):.4f}",
                flush=True,
            )
    finally:
        model.model.eval()
    return losses


@torch.no_grad()
def score_arm(
    model: models.LoadedModel,
    bank: dict[str, ConceptVector],
    prepared: dict[tuple[str, str], PreparedEpisode],
    *,
    arm: str,
    seed: int,
    carriers: tuple[str, ...],
    strength: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in sorted(bank):
        target = bank[concept]
        directions = {"target": target, "random": random_control(target, seed=CONTROL_SEED)}
        conditions = CONDITIONS if strength == FULL_CONDITION_STRENGTH else ("target",)
        for carrier in carriers:
            for episode in exact_episodes(carrier):
                item = prepared[(carrier, episode.cell_id)]
                for condition in conditions:
                    direction = directions[condition]
                    edits = condition_interventions(
                        cast(Any, condition),
                        direction,
                        item.state_positions,
                        episode.state_signs,
                        strength=strength,
                    )
                    with intervene(model, edits, prompt_len=int(item.input_ids.shape[1])):
                        logits = cast(Any, model.model)(item.input_ids).logits
                    row = logits[0, -1].float()
                    probabilities = torch.softmax(row, dim=-1)
                    q_id, k_id = item.label_ids
                    mass = float(probabilities[q_id] + probabilities[k_id])
                    predicted = LABELS[0] if float(row[q_id] - row[k_id]) > 0 else LABELS[1]
                    correct_index = LABELS.index(episode.correct_label)
                    other_index = 1 - correct_index
                    rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "strength": strength,
                            "condition": condition,
                            "concept": concept,
                            "carrier_sha256": hashlib.sha256(carrier.encode()).hexdigest(),
                            "cell_id": episode.cell_id,
                            "order_key": "".join(f"{s:+d}" for s in episode.demo_signs),
                            "positive_label": episode.positive_label,
                            "query_sign": episode.query_sign,
                            "correct_label": episode.correct_label,
                            "predicted_label": predicted,
                            "correct": predicted == episode.correct_label,
                            "signed_margin": float(
                                row[item.label_ids[correct_index]]
                                - row[item.label_ids[other_index]]
                            ),
                            "label_mass": mass,
                            "format_ok": int(row.argmax()) in item.label_ids,
                            "prompt_sha256": item.prompt_sha256,
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
        "--protocol", type=Path, default=ROOT / "results" / "remap_training_protocol_v2.json"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} exists; reproduce into a new path")

    _protocol, protocol_sha = freeze_protocol(args.protocol, args.smoke)
    source_hashes = _source_hashes()
    train_concepts = TRAIN_CONCEPTS[:2] if args.smoke else TRAIN_CONCEPTS
    eval_concepts = EVAL_CONCEPTS[:1] if args.smoke else EVAL_CONCEPTS
    eval_carriers = EVAL_CARRIERS[:1] if args.smoke else EVAL_CARRIERS

    started = time.time()
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    model.model.to(torch.bfloat16)
    object.__setattr__(model, "dtype", torch.bfloat16)

    centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
    center = torch.stack([cv.vector for cv in centering.values()]).mean(dim=0)
    train_bank = _centered_bank(model, train_concepts, center)
    eval_bank = _centered_bank(model, eval_concepts, center)
    for name, bank in (("train", train_bank), ("eval", eval_bank)):
        worst = max_offdiagonal_cosine(bank)
        if worst > MAX_COSINE:
            raise SystemExit(f"{name} bank degenerate: max |cos| {worst:.3f}")
        print(f"{name} bank max |cos| {worst:.3f}", flush=True)

    prepared: dict[tuple[str, str], PreparedEpisode] = {}
    for carrier in (*TRAIN_CARRIERS, *eval_carriers):
        for episode in exact_episodes(carrier):
            prepared[(carrier, episode.cell_id)] = prepare_episode(model, episode)

    rows: list[dict[str, Any]] = []
    for strength in EVAL_STRENGTHS:
        print(f"scoring untrained base at strength {strength}", flush=True)
        rows.extend(
            score_arm(
                model,
                eval_bank,
                prepared,
                arm="base",
                seed=args.seed,
                carriers=eval_carriers,
                strength=strength,
            )
        )
    # LoRA freezes the base weights and ``unload`` strips the adapter without
    # merging, so the base model is untouched between arms. Cloning the whole
    # state dict to "restore" it cost ~6 GB of wired memory and restored nothing.
    # The assumption is checked below instead of held in RAM.
    base_fingerprint = _base_fingerprint(model)

    loss_curves: dict[str, list[float]] = {}
    for arm, fixed in (("fixed", True), ("remap", False)):
        print(f"training {arm} adapter (seed {args.seed})", flush=True)
        if _base_fingerprint(model) != base_fingerprint:
            raise SystemExit("base weights changed between arms; artifact discarded")
        torch.manual_seed(args.seed)
        attach_lora(model, r=LORA_RANK, alpha=LORA_ALPHA)
        for parameter in model.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        loss_curves[arm] = train_adapter(model, train_bank, prepared, fixed=fixed, seed=args.seed)
        for strength in EVAL_STRENGTHS:
            print(f"scoring {arm} at strength {strength}", flush=True)
            rows.extend(
                score_arm(
                    model,
                    eval_bank,
                    prepared,
                    arm=arm,
                    seed=args.seed,
                    carriers=eval_carriers,
                    strength=strength,
                )
            )
        # Unwrap the adapter so the next arm starts from the same base weights.
        model.model = cast(Any, model.model).unload()
        if _base_fingerprint(model) != base_fingerprint:
            raise SystemExit(f"{arm} training altered base weights; artifact discarded")

    if source_hashes != _source_hashes():
        raise SystemExit("source changed during the run; artifact discarded")

    _write_atomic(args.out, [json.dumps(r, sort_keys=True, separators=(",", ":")) for r in rows])
    config = {
        "centering_direction_sha256": _tensor_sha256(center),
        "device": "mps",
        "elapsed_seconds": round(time.time() - started, 1),
        "eval_carriers": list(eval_carriers),
        "eval_concepts": list(eval_concepts),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "model_revision": models.loaded_revision(model),
        "n_steps": {arm: len(curve) for arm, curve in loss_curves.items()},
        "platform": platform.platform(),
        "protocol_sha256": protocol_sha,
        "python": platform.python_version(),
        "smoke": args.smoke,
        "source_files_sha256": source_hashes,
        "source_sha256": json_sha256(source_hashes),
        "torch": torch.__version__,
        "train_carriers": list(TRAIN_CARRIERS),
        "train_concepts": list(train_concepts),
        "train_seed": args.seed,
    }
    manifest = {
        "config": config,
        "config_sha256": json_sha256(config),
        "final_epoch_loss": {arm: curve[-1] for arm, curve in loss_curves.items()},
        "n_rows": len(rows),
        "raw": args.out.name,
        "raw_sha256": _sha256(args.out),
        "schema_version": 1,
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"wrote {args.out} ({len(rows)} rows) in {config['elapsed_seconds']}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
