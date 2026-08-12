"""The equal-or-lower-cost reader, applied to the content task from notes/14.

Pre-run note: ``notes/15-matched-reader-on-content.md``, written before this ran.

``notes/11`` ran this comparison on the polarity task, which ``notes/13`` later
showed collapses to the sign of a projection onto one axis. This runs the identical
comparison on a task that cannot be solved that way, and runs the polarity arm in
the same process so the two are compared internally rather than across runs.

The reader is imported from ``run_matched_reader``; the two-direction intervention
from ``run_content_vs_disturbance``. Neither is reimplemented here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from run_content_vs_disturbance import (
    LAYER,
    MODEL,
    MODEL_REVISION,
    PAIRS,
    STRENGTH,
    matched_strength,
    pair_interventions,
)
from run_matched_reader import _read

from introspect import models
from introspect.codebook_icl import (
    CONFIRM_CONCEPTS,
    CONFIRM_VISIBLE_SAMPLES,
    LABELS,
    exact_episodes,
    prepare_episode,
)
from introspect.concepts import ConceptVector, build_bank, pairwise_cosines, random_control
from introspect.hooks import capture, intervene
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

READERS = ("centroid_euclidean", "centroid_cosine", "shuffled_labels")
ARMS = ("content", "polarity", "random_polarity", "polarity_weak")

#: notes/08 evaluated the trained adapters here; the base model is at 0.500.
WEAK_STRENGTH = 0.15


def signed_overlap(bank: dict[str, ConceptVector]) -> dict[str, float]:
    """Signed pairwise cosines. run_bank_audit took abs() before counting positives,
    which made its ``n_positive`` vacuous. This is the corrected measurement."""
    values = list(pairwise_cosines(bank).values())
    t = torch.tensor(values)
    return {
        "n": len(values),
        "n_positive_signed": int((t > 0).sum()),
        "mean_signed": float(t.mean()),
        "mean_abs": float(t.abs().mean()),
        "min_signed": float(t.min()),
        "max_signed": float(t.max()),
    }


@torch.no_grad()
def episode_row(
    model: models.LoadedModel,
    prepared: object,
    positive: ConceptVector,
    negative: ConceptVector,
    *,
    strength: float,
    seed: int,
) -> dict[str, object]:
    """Score the model and every reader from one forward pass."""
    p = prepared
    signs = p.episode.state_signs  # type: ignore[attr-defined]
    interventions = pair_interventions(
        positive,
        negative,
        p.state_positions,  # type: ignore[attr-defined]
        signs,
        strength=strength,
    )
    ids = p.input_ids  # type: ignore[attr-defined]
    with (
        intervene(model, interventions, prompt_len=int(ids.shape[1])),
        capture(model, [LAYER]) as store,
    ):
        logits = model.forward_logits(ids)[0, -1].float()

    states = store.acts[LAYER][0][0, list(p.state_positions)].float().cpu()  # type: ignore[attr-defined]
    if states.shape[0] != 5:
        raise ValueError(f"expected 5 captured states, got {states.shape[0]}")

    label_ids = p.label_ids  # type: ignore[attr-defined]
    selected = logits[torch.tensor(label_ids, device=logits.device)]
    predicted = LABELS[int(selected.argmax())]
    correct_label = p.episode.correct_label  # type: ignore[attr-defined]
    query_sign = p.episode.query_sign  # type: ignore[attr-defined]

    row: dict[str, object] = {
        "model_predicted": predicted,
        "model_correct": predicted == correct_label,
        "model_format_ok": int(logits.argmax()) in set(label_ids),
        "query_sign": query_sign,
    }
    for reader in READERS:
        row[f"reader_{reader}_correct"] = _read(states, signs, reader, seed) == query_sign
    return row


def _paired(rows: list[dict[str, object]], reader: str) -> dict[str, int]:
    key = f"reader_{reader}_correct"
    table = {"both": 0, "model_only": 0, "reader_only": 0, "neither": 0}
    for r in rows:
        m, d = bool(r["model_correct"]), bool(r[key])
        table["both" if m and d else "model_only" if m else "reader_only" if d else "neither"] += 1
    return table


def _self_check() -> None:
    """Four demonstrations, two per class, with the query on a known side."""
    states = torch.zeros(5, 8)
    states[0][0], states[1][0] = 1.0, 1.0  # two positives
    states[2][0], states[3][0] = -1.0, -1.0  # two negatives
    states[4][0] = 0.9  # query sits with the positives
    assert _read(states, (1, 1, -1, -1, 1), "centroid_euclidean", 0) == 1
    states[4][0] = -0.9
    assert _read(states, (1, 1, -1, -1, -1), "centroid_euclidean", 0) == -1


def run(args: argparse.Namespace) -> None:
    _self_check()
    out = args.out
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    for path in (out, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    started = time.time()
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

        centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
        center = torch.stack([cv.vector for cv in centering.values()]).mean(dim=0)
        raw = build_bank(model, LAYER, list(CONFIRM_CONCEPTS), center=False)
        bank = {
            name: ConceptVector(name=name, layer=LAYER, vector=cv.vector - center)
            for name, cv in raw.items()
        }
        print("bank built", flush=True)

        pairs = PAIRS[:1] if args.smoke else PAIRS
        carriers = CONFIRM_VISIBLE_SAMPLES[:1] if args.smoke else CONFIRM_VISIBLE_SAMPLES

        rows: list[dict[str, object]] = []
        for carrier in carriers:
            episodes = exact_episodes(carrier)
            if args.smoke:
                episodes = episodes[:2]
            prepared_all = [prepare_episode(model, e) for e in episodes]
            carrier_sha = __import__("hashlib").sha256(carrier.encode()).hexdigest()[:16]

            for name_a, name_b in pairs:
                a, b = bank[name_a], bank[name_b]
                negated = ConceptVector(name=f"-{name_a}", layer=LAYER, vector=-a.vector)
                r = random_control(a, seed=0)
                r_neg = ConceptVector(name=f"-{r.name}", layer=LAYER, vector=-r.vector)
                plans = {
                    "content": (a, b, matched_strength(a.vector, b.vector)),
                    "polarity": (a, negated, matched_strength(a.vector, negated.vector)),
                    # The arm that tests whether the adaptive reader keeps the
                    # generality notes/13 credited to training.
                    "random_polarity": (r, r_neg, matched_strength(r.vector, r_neg.vector)),
                    # notes/08's weak regime: base model blind at 0.500, trained 0.79-0.86.
                    "polarity_weak": (a, negated, WEAK_STRENGTH),
                }
                for arm, (pos, neg, strength) in plans.items():
                    for seed, prepared in enumerate(prepared_all):
                        rows.append(
                            {
                                "arm": arm,
                                "pair": f"{name_a}|{name_b}",
                                "carrier_sha": carrier_sha,
                                "cell_id": prepared.episode.cell_id,
                                "strength": strength,
                                **episode_row(
                                    model, prepared, pos, neg, strength=strength, seed=seed
                                ),
                            }
                        )
                print(f"{carrier_sha}: {name_a}|{name_b}", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        arms: dict[str, object] = {}
        for arm in ARMS:
            subset = [r for r in rows if r["arm"] == arm]
            if not subset:
                continue
            readers = {
                reader: sum(bool(r[f"reader_{reader}_correct"]) for r in subset) / len(subset)
                for reader in READERS
            }
            model_acc = sum(bool(r["model_correct"]) for r in subset) / len(subset)
            arms[arm] = {
                "n": len(subset),
                "model_accuracy": model_acc,
                "reader_accuracy": readers,
                "model_minus_best_fair_reader": model_acc
                - max(readers["centroid_euclidean"], readers["centroid_cosine"]),
                "paired_vs_centroid_euclidean": _paired(subset, "centroid_euclidean"),
                "format_rate": sum(bool(r["model_format_ok"]) for r in subset) / len(subset),
                "by_pair": {
                    pair: {
                        "model": sum(bool(r["model_correct"]) for r in subset if r["pair"] == pair)
                        / len([r for r in subset if r["pair"] == pair]),
                        "reader": sum(
                            bool(r["reader_centroid_euclidean_correct"])
                            for r in subset
                            if r["pair"] == pair
                        )
                        / len([r for r in subset if r["pair"] == pair]),
                    }
                    for pair in sorted({str(r["pair"]) for r in subset})
                },
            }

        summary = {
            "what_this_is": (
                "The equal-or-lower-cost reader from notes/11, applied to the content "
                "task from notes/14, with the polarity task run in the same process."
            ),
            "criterion": "https://arxiv.org/abs/2508.14802",
            "smoke": bool(args.smoke),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "base_strength": STRENGTH,
            "weak_strength": WEAK_STRENGTH,
            "arms": arms,
            "corrected_signed_overlap": {
                "note": (
                    "run_bank_audit.py applied abs() before counting positives, making "
                    "its n_positive vacuous. These are the signed values."
                ),
                "eval_bank": signed_overlap(bank),
            },
            "elapsed_seconds": round(time.time() - started, 1),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"wrote {out} and {summary_path}", flush=True)
    finally:
        model.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("results/matched_reader_content_v1_raw.jsonl")
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
