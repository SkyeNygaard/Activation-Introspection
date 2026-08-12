"""Is the per-position intervention builder equivalent to notes/14's?

``run_heldout_semantic``'s ``same_exemplar`` anchor scored 0.521 twin-pair where
``14``'s content arm scored 0.799. Two explanations, and they demand opposite
responses:

1. the new builder edits the residual stream differently -- an apparatus bug, and
   nothing in the held-out run means anything;
2. the new exemplars are simply harder to report than ``14``'s bank, in which
   case the machinery is sound and the anchor shortfall is about word choice.

This separates them. Same concepts, same episodes, same strength, both builders,
scored one against the other. ``exemplar_interventions`` with a plan that repeats
one exemplar per class should be *mathematically* the same edit as
``pair_interventions`` -- five one-position edits instead of two multi-position
edits. If predictions agree episode for episode, the builder is exonerated.

    uv run python scripts/diagnose_intervention_equivalence.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from run_content_vs_disturbance import (
    LAYER,
    MODEL,
    MODEL_REVISION,
    PAIRS,
    matched_strength,
    pair_interventions,
    twin_pair,
)
from run_heldout_semantic import exemplar_interventions

from introspect import models
from introspect.codebook_icl import (
    CONFIRM_CONCEPTS,
    CONFIRM_VISIBLE_SAMPLES,
    LABELS,
    exact_episodes,
    prepare_episode,
)
from introspect.concepts import ConceptVector, build_bank
from introspect.hooks import Intervention, intervene
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

OUT = Path("results/intervention_equivalence_v1_summary.json")

#: notes/14's content arm on this bank, for reference.
NOTES_14_CONTENT_TWIN_PAIR = 0.799


@torch.no_grad()
def score(model: models.LoadedModel, prepared: object, ivs: list[Intervention]) -> str:
    p = prepared
    ids = p.input_ids  # type: ignore[attr-defined]
    with intervene(model, ivs, prompt_len=int(ids.shape[1])):
        logits = model.forward_logits(ids)[0, -1].float()
    label_ids = p.label_ids  # type: ignore[attr-defined]
    selected = logits[torch.tensor(label_ids, device=logits.device)]
    return LABELS[int(selected.argmax())]


def main() -> None:
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
            n: ConceptVector(name=n, layer=LAYER, vector=cv.vector - center)
            for n, cv in raw.items()
        }

        carrier = CONFIRM_VISIBLE_SAMPLES[0]
        episodes = exact_episodes(carrier)
        prepared_all = [prepare_episode(model, e) for e in episodes]

        rows: list[dict[str, object]] = []
        for name_a, name_b in PAIRS:
            a, b = bank[name_a], bank[name_b]
            strength = matched_strength(a.vector, b.vector)
            plan = {
                "demo_a": [name_a, name_a],
                "demo_b": [name_b, name_b],
                "query_a": name_a,
                "query_b": name_b,
            }
            for prepared in prepared_all:
                ep = prepared.episode
                positions, signs = prepared.state_positions, ep.state_signs
                old = score(
                    model,
                    prepared,
                    pair_interventions(a, b, positions, signs, strength=strength),
                )
                new = score(
                    model,
                    prepared,
                    exemplar_interventions(bank, plan, positions, signs, strength=strength),
                )
                rows.append(
                    {
                        "pair": f"{name_a}|{name_b}",
                        "carrier_sha": "c0",
                        "cell_base": ep.cell_id.rsplit("q", 1)[0],
                        "old_predicted": old,
                        "new_predicted": new,
                        "agree": old == new,
                        "old_correct": old == ep.correct_label,
                        "new_correct": new == ep.correct_label,
                    }
                )

        agreement = sum(bool(r["agree"]) for r in rows) / len(rows)
        old_tp = twin_pair([dict(r, correct=r["old_correct"]) for r in rows])
        new_tp = twin_pair([dict(r, correct=r["new_correct"]) for r in rows])
        summary = {
            "what_this_is": __doc__.strip().splitlines()[0],
            "n_episodes": len(rows),
            "agreement_rate": agreement,
            "old_builder_twin_pair": old_tp,
            "new_builder_twin_pair": new_tp,
            "notes_14_content_twin_pair": NOTES_14_CONTENT_TWIN_PAIR,
            "elapsed_seconds": round(time.time() - started, 1),
            "verdict": (
                "builders equivalent; anchor shortfall is the exemplars"
                if agreement == 1.0
                else "builders differ; the held-out run has an apparatus bug"
            ),
        }
        OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        del model
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
