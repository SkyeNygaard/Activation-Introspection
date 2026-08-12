"""Two concepts instead of one concept and its negation: content or disturbance?

Pre-run note: ``notes/14-content-versus-disturbance.md``, written before this ran.

``notes/13`` showed the existing reporting task collapses to the sign of a
projection onto one shared axis, so it never required knowing *which* concept was
injected. This changes the two classes from ``+v``/``-v`` to ``v_A``/``v_B`` and
changes nothing else, keeping the property that makes the design worth anything:
query twins are byte-identical in visible text with opposite correct labels, so an
input-only learner is pinned at 0.500 by construction.

``codebook_icl`` is imported, not modified -- frozen protocols record its hash.
The only new logic is a two-direction intervention builder and the per-pair
strength that equalizes the distance between the two classes across arms.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from introspect import models
from introspect.codebook_icl import (
    CONFIRM_CONCEPTS,
    CONFIRM_VISIBLE_SAMPLES,
    LABELS,
    Episode,
    exact_episodes,
    prepare_episode,
    score_episode,
)
from introspect.concepts import ConceptVector, build_bank, random_control
from introspect.hooks import Intervention, intervene
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LAYER = 9
STRENGTH = 1.0

#: Four disjoint pairs from the confirmation bank behind the frozen 0.891.
PAIRS = (
    ("garden", "camera"),
    ("train", "banana"),
    ("eagle", "library"),
    ("hammer", "island"),
)

ARMS = ("polarity", "content", "random_pair", "query_only", "clean")


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-8)


def matched_strength(a: torch.Tensor, b: torch.Tensor, *, base: float = STRENGTH) -> float:
    """Strength making ``a``/``b`` as far apart as opposite poles would be.

    Distance between classes is ``strength * |h| * ||unit(a) - unit(b)||``.
    Opposite poles give 2.0, so this returns exactly ``base`` for them and scales
    every other pair up to match.
    """
    return float(2.0 * base / ((_unit(a) - _unit(b)).norm() + 1e-8))


def pair_interventions(
    positive: ConceptVector,
    negative: ConceptVector,
    positions: tuple[int, ...],
    signs: tuple[int, ...],
    *,
    strength: float,
    query_only: bool = False,
) -> list[Intervention]:
    """``+1`` positions receive ``positive``; ``-1`` positions receive ``negative``.

    Unlike ``codebook_icl.condition_interventions`` the two classes are two
    independent directions rather than one direction and its negation.
    """
    if len(positions) != len(signs):
        raise ValueError("each state sign needs one intervention position")
    if query_only:
        positions, signs = positions[-1:], signs[-1:]
    out = []
    for sign, direction in ((1, positive), (-1, negative)):
        selected = [p for p, state in zip(positions, signs, strict=True) if state == sign]
        if selected:
            out.append(
                Intervention(
                    layer=LAYER,
                    direction=direction.vector,
                    strength=strength,
                    positions=selected,
                    per_position=True,
                    label=f"{direction.name}:{sign:+d}",
                )
            )
    return out


@torch.no_grad()
def score_pair(
    model: models.LoadedModel,
    prepared: object,
    positive: ConceptVector,
    negative: ConceptVector,
    *,
    strength: float,
    query_only: bool = False,
) -> dict[str, object]:
    """Score one episode under a two-direction edit. Mirrors ``score_episode``."""
    p = prepared  # typed loosely: PreparedEpisode is frozen and only read here
    interventions = pair_interventions(
        positive,
        negative,
        p.state_positions,  # type: ignore[attr-defined]
        p.episode.state_signs,  # type: ignore[attr-defined]
        strength=strength,
        query_only=query_only,
    )
    ids = p.input_ids  # type: ignore[attr-defined]
    with intervene(model, interventions, prompt_len=int(ids.shape[1])):
        logits = model.forward_logits(ids)[0, -1].float()
    label_ids = p.label_ids  # type: ignore[attr-defined]
    candidates = torch.tensor(label_ids, device=logits.device)
    selected = logits[candidates]
    predicted = LABELS[int(selected.argmax())]
    return {
        "predicted_label": predicted,
        "correct": predicted == p.episode.correct_label,  # type: ignore[attr-defined]
        "label_mass": float(
            torch.logsumexp(selected, 0).sub(torch.logsumexp(logits, 0)).exp()
        ),
        "format_ok": int(logits.argmax()) in set(label_ids),
    }


def twin_pair(rows: list[dict[str, object]]) -> float:
    """A cell counts only if both query states are read correctly."""
    cells: dict[tuple[str, str, str], list[bool]] = {}
    for r in rows:
        key = (str(r["pair"]), str(r["carrier_sha"]), str(r["cell_base"]))
        cells.setdefault(key, []).append(bool(r["correct"]))
    complete = [v for v in cells.values() if len(v) == 2]
    if not complete:
        return float("nan")
    return sum(all(v) for v in complete) / len(complete)


def _self_check() -> None:
    """Matched strength on vectors whose answers are known."""
    a = torch.randn(128)
    assert abs(matched_strength(a, -a) - 1.0) < 1e-4, "opposite poles must return base"
    e1, e2 = torch.zeros(128), torch.zeros(128)
    e1[0], e2[1] = 1.0, 1.0
    # orthogonal unit vectors are sqrt(2) apart, so strength must be 2/sqrt(2)
    assert abs(matched_strength(e1, e2) - 2.0 / (2**0.5)) < 1e-4
    rows = [
        {"pair": "p", "carrier_sha": "c", "cell_base": "x", "correct": True},
        {"pair": "p", "carrier_sha": "c", "cell_base": "x", "correct": False},
    ]
    assert twin_pair(rows) == 0.0
    rows[1]["correct"] = True
    assert twin_pair(rows) == 1.0


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
                r_a = random_control(a, seed=0)
                r_b = random_control(b, seed=1)
                plans = {
                    "polarity": (a, negated, matched_strength(a.vector, negated.vector), False),
                    "content": (a, b, matched_strength(a.vector, b.vector), False),
                    "random_pair": (r_a, r_b, matched_strength(r_a.vector, r_b.vector), False),
                    "query_only": (a, b, matched_strength(a.vector, b.vector), True),
                }
                for arm, (pos, neg, strength, q_only) in plans.items():
                    for prepared in prepared_all:
                        ep: Episode = prepared.episode
                        result = score_pair(
                            model, prepared, pos, neg, strength=strength, query_only=q_only
                        )
                        rows.append(
                            {
                                "arm": arm,
                                "pair": f"{name_a}|{name_b}",
                                "carrier_sha": carrier_sha,
                                "cell_id": ep.cell_id,
                                "cell_base": ep.cell_id.rsplit("q", 1)[0],
                                "query_sign": ep.query_sign,
                                "strength": strength,
                                **result,
                            }
                        )
                print(f"{carrier_sha}: {name_a}|{name_b}", flush=True)

            # Clean does not depend on the concept pair, so it runs once per carrier.
            for prepared in prepared_all:
                score = score_episode(model, prepared, "clean", None, strength=STRENGTH)
                rows.append(
                    {
                        "arm": "clean",
                        "pair": "-",
                        "carrier_sha": carrier_sha,
                        "cell_id": prepared.episode.cell_id,
                        "cell_base": prepared.episode.cell_id.rsplit("q", 1)[0],
                        "query_sign": prepared.episode.query_sign,
                        "strength": 0.0,
                        "predicted_label": score.predicted_label,
                        "correct": score.correct,
                        "label_mass": score.label_mass,
                        "format_ok": score.format_ok,
                    }
                )

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        arms: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            subset = [r for r in rows if r["arm"] == arm]
            if not subset:
                continue
            arms[arm] = {
                "accuracy": sum(bool(r["correct"]) for r in subset) / len(subset),
                "twin_pair": twin_pair(subset),
                "format_rate": sum(bool(r["format_ok"]) for r in subset) / len(subset),
                "mean_label_mass": sum(float(r["label_mass"]) for r in subset) / len(subset),  # type: ignore[arg-type]
                "n": len(subset),
            }
        by_pair = {
            arm: {
                pair: sum(
                    bool(r["correct"]) for r in rows if r["arm"] == arm and r["pair"] == pair
                )
                / max(1, len([r for r in rows if r["arm"] == arm and r["pair"] == pair]))
                for pair in sorted({str(r["pair"]) for r in rows if r["arm"] == arm})
            }
            for arm in ("polarity", "content", "random_pair")
        }

        summary = {
            "what_this_is": (
                "Two concepts instead of one concept and its negation. Separation "
                "between the two classes is equalized across arms by construction."
            ),
            "smoke": bool(args.smoke),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "base_strength": STRENGTH,
            "pairs": [f"{a}|{b}" for a, b in pairs],
            "arms": arms,
            "accuracy_by_pair": by_pair,
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
        "--out", type=Path, default=Path("results/content_vs_disturbance_v1_raw.jsonl")
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
