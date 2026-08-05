"""Study 1 runner: causal use of a retained activation trace.

Injects a concept direction into a neutral carrier's residual stream, removes
the hook, then reveals a freshly sampled concept->label codebook and scores the
label. Because the codebook does not exist while the edit is live, the edit
cannot have promoted the correct answer token.

Writes one raw JSONL row per trial plus a checksummed summary with provenance.
Aggregates are always regenerated from the raw rows; nothing is hand-edited.

    uv run python scripts/run_retained_trace.py --split dev --layers 4,8,12,16,20
    uv run python scripts/run_retained_trace.py --split test --layers 12 --strength 4.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect import concepts as concept_mod
from introspect import models, retained

ARMS = ["clean", "sham", "target", "random", "shuffled"]


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def model_revision(repo: str) -> str:
    """Immutable snapshot hash of the local weights, when resolvable."""
    try:
        from huggingface_hub import HfApi

        return str(HfApi().model_info(repo).sha)
    except Exception:
        return "unknown"


@dataclass
class Row:
    inject_layer: int
    strength: float
    carrier_id: int
    codebook_id: int
    concept: str
    arm: str
    correct_label: str
    pred_label: str
    correct: bool
    p_correct: float
    format_ok: bool
    carrier_kl: float
    codebook_digest: str


def run(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]
    bank_names = retained.DEV_CONCEPTS if args.split == "dev" else retained.TEST_CONCEPTS
    layers = [int(x) for x in args.layers.split(",")]
    strengths = [float(x) for x in str(args.strength).split(",")]

    m = models.load(args.model)
    warn = models.memory_warning(args.model)
    if warn:
        print(f"warning: {warn}", file=sys.stderr)

    labels = retained.single_token_labels(m)[: len(bank_names)]
    if len(labels) < len(bank_names):
        raise SystemExit("not enough single-token labels for this concept bank")
    label_ids = retained.label_token_ids(m, labels)
    codebooks = retained.balanced_codebooks(bank_names, labels)
    readout_layers = list(range(m.n_layers))

    print(
        f"model={m.name} split={args.split} concepts={len(bank_names)} "
        f"labels={labels} chance={1 / len(bank_names):.3f}",
        flush=True,
    )

    rows: list[Row] = []
    act_index: list[dict[str, Any]] = []
    act_store: list[torch.Tensor] = []
    t0 = time.time()

    for layer in layers:
        bank = concept_mod.build_bank(m, layer, list(bank_names))
        max_cos = concept_mod.max_offdiagonal_cosine(bank)
        if max_cos > args.max_cosine:
            print(
                f"  layer {layer}: SKIP, concept vectors near-collinear "
                f"(max |cos| = {max_cos:.3f} > {args.max_cosine})",
                flush=True,
            )
            continue

        clean_logits_by_carrier: dict[int, torch.Tensor] = {}

        def emit(
            cc: retained.CarrierCache,
            splits: list[retained.Split],
            *,
            arm: str,
            strength: float,
            carrier_id: int,
            scored_concepts: list[str],
            clean_logits: dict[int, torch.Tensor],
            layer: int = layer,
        ) -> None:
            kl = retained.carrier_kl(clean_logits[carrier_id], cc.carrier_logits)
            for cb_id, (cb, split) in enumerate(zip(codebooks, splits, strict=True)):
                probs, pred, fmt = retained.query_cache(m, cc, split.stage2, cb, label_ids)
                for concept in scored_concepts:
                    correct_label = cb.mapping[concept]
                    rows.append(
                        Row(
                            inject_layer=layer,
                            strength=strength,
                            carrier_id=carrier_id,
                            codebook_id=cb_id,
                            concept=concept,
                            arm=arm,
                            correct_label=correct_label,
                            pred_label=pred,
                            correct=pred == correct_label,
                            p_correct=probs[correct_label],
                            format_ok=fmt,
                            carrier_kl=kl,
                            codebook_digest=cb.digest(),
                        )
                    )

        for carrier_id, carrier_text in enumerate(retained.CARRIER_FAMILIES):
            splits = [retained.split_prompt(m, carrier_text, cb) for cb in codebooks]
            stage1 = splits[0].stage1
            for s in splits[1:]:
                if not torch.equal(s.stage1, stage1):
                    raise RuntimeError("carrier stage-1 ids differ across codebooks")
            positions = retained.carrier_positions(int(stage1.shape[1]))

            # --- concept-independent arms, scored against every concept -----
            # Neither depends on strength: clean registers no hook, sham runs
            # the identical hook at strength zero.
            for arm in ("clean", "sham"):
                ivs = retained.build_arm(arm, bank[bank_names[0]], layer, 0.0, positions)
                cc = retained.build_carrier_cache(m, stage1, interventions=ivs)
                if arm == "clean":
                    clean_logits_by_carrier[carrier_id] = cc.carrier_logits
                emit(
                    cc,
                    splits,
                    arm=arm,
                    strength=0.0,
                    carrier_id=carrier_id,
                    scored_concepts=list(bank_names),
                    clean_logits=clean_logits_by_carrier,
                )

            # --- ceiling arm: the concept stated in plain text ---------------
            # Not damage-matched and not a control -- it is the reference for
            # what "the model can do this task at all" looks like. A null in the
            # injected arms is only interpretable if this arm is well above
            # chance.
            for concept in bank_names:
                nat_text = retained.natural_carrier(carrier_text, concept)
                nat_splits = [retained.split_prompt(m, nat_text, cb) for cb in codebooks]
                cc = retained.build_carrier_cache(m, nat_splits[0].stage1)
                emit(
                    cc,
                    nat_splits,
                    arm="natural",
                    strength=0.0,
                    carrier_id=carrier_id,
                    scored_concepts=[concept],
                    clean_logits=clean_logits_by_carrier,
                )

            # --- concept-dependent arms -------------------------------------
            for strength in strengths:
                for concept in bank_names:
                    vec = bank[concept]
                    arm_vectors = {
                        "target": vec,
                        "random": concept_mod.random_control(vec, seed=args.seed),
                        "shuffled": concept_mod.shuffled_control(vec, seed=args.seed),
                    }
                    for arm, arm_vec in arm_vectors.items():
                        ivs = retained.build_arm(arm, arm_vec, layer, strength, positions)
                        cc = retained.build_carrier_cache(
                            m, stage1, interventions=ivs, capture_layers=readout_layers
                        )
                        act_index.append(
                            {
                                "row": len(act_store),
                                "inject_layer": layer,
                                "strength": strength,
                                "carrier_id": carrier_id,
                                "concept": concept,
                                "arm": arm,
                            }
                        )
                        act_store.append(
                            torch.stack([cc.acts[r] for r in readout_layers]).to(torch.float16)
                        )
                        emit(
                            cc,
                            splits,
                            arm=arm,
                            strength=strength,
                            carrier_id=carrier_id,
                            scored_concepts=[concept],
                            clean_logits=clean_logits_by_carrier,
                        )

        base = [
            r for r in rows if r.inject_layer == layer and r.arm in ("clean", "sham", "natural")
        ]
        floor = {
            a: _mean([r.correct for r in base if r.arm == a]) for a in ("clean", "sham", "natural")
        }
        for strength in strengths:
            done = [
                r
                for r in rows
                if r.inject_layer == layer
                and r.strength == strength
                and r.arm not in ("clean", "sham")
            ]
            by_arm = {a: _mean([r.correct for r in done if r.arm == a]) for a in ARMS[2:]}
            kl_t = _mean([r.carrier_kl for r in done if r.arm == "target"])
            fmt_t = _mean([r.format_ok for r in done if r.arm == "target"])
            print(
                f"  L{layer:>2} a={strength:<4g} cos={max_cos:.2f} kl={kl_t:6.2f} "
                f"fmt={fmt_t:.2f} | "
                f"nat={floor['natural']:.3f} clean={floor['clean']:.3f} "
                f"sham={floor['sham']:.3f} "
                + " ".join(f"{a}={by_arm[a]:.3f}" for a in ARMS[2:])
                + f"  [{time.time() - t0:.0f}s]",
                flush=True,
            )

    raw_path = Path(args.raw)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r.__dict__) + "\n")
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    if act_store:
        act_path = raw_path.with_suffix(".acts.pt")
        torch.save({"index": act_index, "acts": torch.stack(act_store)}, act_path)
    else:
        act_path = None

    summary = {
        "estimand": "post_cache_codebook_label_probability",
        "schedule": "transient_carrier_injection_hook_removed_before_codebook",
        "split": args.split,
        "concepts": list(bank_names),
        "labels": labels,
        "chance": 1 / len(bank_names),
        "layers": layers,
        "strengths": strengths,
        "carriers": len(retained.CARRIER_FAMILIES),
        "codebooks": len(codebooks),
        "n_rows": len(rows),
        "raw": raw_path.name,
        "raw_sha256": raw_sha,
        "activations": act_path.name if act_path else None,
        "model": m.name,
        "model_revision": model_revision(m.name),
        "device": str(m.device),
        "dtype": str(m.dtype),
        "git_commit": git_commit(root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "query_sha256": hashlib.sha256(retained.QUERY.encode()).hexdigest()[:16],
        "carrier_sha256": hashlib.sha256("|".join(retained.CARRIER_FAMILIES).encode()).hexdigest()[
            :16
        ],
        "seed": args.seed,
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {len(rows)} rows -> {raw_path}\nsummary -> {args.summary}")


def _mean(xs: list[Any]) -> float:
    vals = [float(x) for x in xs]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-0.5b")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--layers", default="4,8,12,16,20")
    ap.add_argument("--strength", default="4.0", help="comma-separated for a sweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cosine", type=float, default=0.5)
    ap.add_argument("--raw", default="results/retained_dev_raw.jsonl")
    ap.add_argument("--summary", default="results/retained_dev_summary.json")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
