"""notes/43 -- the comparator ladder, measured as one thing instead of assembled.

Pre-run note: ``notes/43-the-ladder-at-proper-power.md``, written before this ran.

notes/20's ladder takes its four numbers from three separate runs at three settings,
with the lens tier hard-coded as a constant. One of the four was a number notes/21
had already withdrawn. This measures every tier in one process, from the same
injected state, across three injection strengths and two elicitations, on eleven
carriers instead of three.

Nothing is reimplemented: the injection site and question come from
``run_zero_shot_identify``, the elicitation wordings from ``run_elicitation_sweep``,
the free-form injection context and reader prompt from ``run_comparator_tiers``, and
the lens from ``run_lens_depth``. This script only arranges them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
from run_comparator_tiers import READER_ASK, inject_prompt_only
from run_elicitation_sweep import ELICITATIONS
from run_lens_depth import lens_all
from run_zero_shot_identify import (
    ANSWER_PREFIX,
    CHANCE,
    LAYER,
    MARKER,
    MODEL,
    MODEL_REVISION,
    first_token_ids,
    question,
)

from introspect import models
from introspect.codebook_icl import CONFIRM_CONCEPTS, CONFIRM_VISIBLE_SAMPLES
from introspect.concepts import ConceptVector, build_bank
from introspect.grading import grade_free_form
from introspect.hooks import Intervention, capture, intervene
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

STRENGTHS = (1.0, 2.0, 4.0)
#: notes/20's prompt, and notes/21's best. The two ends of the elicitation range.
ELICITS = ("describe_gagged", "sensory")
MAX_NEW = 40

#: The gate. notes/21 reproduced notes/20 at exactly this value.
ANCHOR_CELL = ("describe_gagged", 2.0)
ANCHOR_EXPECTED = 0.292
ANCHOR_TOLERANCE = 0.12

#: Eight sentences written by a model blind to any hypothesis, generated for
#: notes/41 and reused here so the ladder is not measured on three sentences
#: from one author. The marker suffix is what the injection site keys on.
BLIND_CARRIERS_PATH = Path("results/blind_carriers_v1.json")
N_BLIND = 8


def carriers_for(smoke: bool) -> list[dict[str, str]]:
    originals = [{"text": c, "source": "original"} for c in CONFIRM_VISIBLE_SAMPLES]
    if smoke:
        return originals[:1]
    blind = json.loads(BLIND_CARRIERS_PATH.read_text())["sentences"][:N_BLIND]
    return originals + [
        {"text": f"{s.rstrip()} Hidden state marker: {MARKER}", "source": "blind"} for s in blind
    ]


def run(args: argparse.Namespace) -> None:
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
        rawbank = build_bank(model, LAYER, list(CONFIRM_CONCEPTS), center=False)
        bank = {
            n: ConceptVector(name=n, layer=LAYER, vector=cv.vector - center)
            for n, cv in rawbank.items()
        }
        options = list(CONFIRM_CONCEPTS)
        listing = ", ".join(options)
        concepts = options[:2] if args.smoke else options
        carriers = carriers_for(args.smoke)
        strengths = STRENGTHS[1:2] if args.smoke else STRENGTHS
        elicits = ELICITS[:1] if args.smoke else ELICITS
        print(f"{len(carriers)} carriers x {len(concepts)} concepts x "
              f"{len(strengths)} strengths x {len(elicits)} elicitations", flush=True)

        rows: list[dict[str, Any]] = []
        for carrier in carriers:
            text = carrier["text"]
            choice_prompt = model.chat(f"{text}\n\n" + question(options), ANSWER_PREFIX)
            choice_ids = model.encode(choice_prompt)
            option_ids = first_token_ids(model, choice_prompt, options)
            # The lens reads the model's own vocabulary head, so score the same
            # eight option words through it and take the best.
            lens_option_ids = option_ids

            def marker_pos(ids: torch.Tensor) -> int:
                for pos, tok in enumerate(ids[0].tolist()):
                    if MARKER in model.tokenizer.decode([tok]):
                        return pos
                raise ValueError("marker token not found")

            choice_marker = marker_pos(choice_ids)

            for elicit in elicits:
                free_prompt = model.chat(f"{text}\n\n{ELICITATIONS[elicit]}")
                free_ids = model.encode(free_prompt)
                free_marker = marker_pos(free_ids)

                for strength in strengths:
                    for concept in concepts:
                        vec = bank[concept].vector
                        edit = Intervention(
                            layer=LAYER, direction=vec, strength=strength,
                            positions=[choice_marker], per_position=True, label=concept,
                        )
                        # Both tiers off ONE forward pass. intervene is entered
                        # before capture, so the captured state is post-edit --
                        # the same state the model itself just read. Reconstructing
                        # it by hand would duplicate Intervention's scaling rule and
                        # is exactly the kind of quiet divergence this repo has been
                        # bitten by; the hook is the single source of truth.
                        with intervene(model, [edit], prompt_len=int(choice_ids.shape[1])), \
                                capture(model, [LAYER]) as store:
                            logits = model.forward_logits(choice_ids)[0, -1].float().cpu()
                        sel = logits[option_ids]
                        pick = options[int(sel.argmax())]

                        # Tier: a lens on that identical state, no labels, no prompt.
                        edited_state = store.acts[LAYER][0][0, choice_marker].float().cpu()
                        lens_logits = lens_all(model, edited_state)
                        lens_pick = options[int(lens_logits[lens_option_ids].argmax())]

                        # Tier: the model's words, then a reader given only those words.
                        # BUG in v1, fixed here: inject_prompt_only takes no strength
                        # and used run_comparator_tiers' hard-coded 2.0, so all three
                        # "strength" cells of this tier were the same run. Verified by
                        # 176/176 report strings being identical across strengths.
                        with inject_prompt_only(model, bank[concept], free_marker,
                                                strength=strength):
                            gen = model.generate_ids(free_ids, max_new_tokens=MAX_NEW,
                                                     do_sample=False)
                        report = model.tokenizer.decode(
                            gen[0][int(free_ids.shape[1]):], skip_special_tokens=True).strip()
                        reader_prompt = model.chat(
                            READER_ASK.format(report=report, options=listing), ANSWER_PREFIX)
                        r_ids = first_token_ids(model, reader_prompt, options)
                        r_logits = model.forward_logits(model.encode(reader_prompt))[0, -1].float()
                        reader_pick = options[int(r_logits[r_ids].argmax().cpu())]

                        rows.append({
                            "carrier_source": carrier["source"], "carrier": text[:28],
                            "elicitation": elicit, "strength": strength, "concept": concept,
                            "model_pick": pick, "model_correct": pick == concept,
                            "lens_pick": lens_pick, "lens_correct": lens_pick == concept,
                            "reader_pick": reader_pick, "reader_correct": reader_pick == concept,
                            "report": report,
                            "report_names_target": grade_free_form(report, concept),
                            "report_leaks_marker": MARKER in report,
                        })
                print(f"  {carrier['source']:8} {elicit:16} ({time.time()-started:.0f}s, "
                      f"{len(rows)} rows)", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")

        def acc(sub: list[dict[str, Any]], key: str) -> float:
            return sum(bool(r[key]) for r in sub) / len(sub) if sub else float("nan")

        cells: dict[str, Any] = {}
        for elicit in elicits:
            for strength in strengths:
                sub = [r for r in rows if r["elicitation"] == elicit and r["strength"] == strength]
                cells[f"{elicit}@{strength}"] = {
                    "n": len(sub),
                    "T0_prompt_only_by_construction": CHANCE,
                    "T1_reader_on_model_words": acc(sub, "reader_correct"),
                    "model_forced_choice": acc(sub, "model_correct"),
                    "T2_lens_on_activations": acc(sub, "lens_correct"),
                    "report_names_target_rate": acc(sub, "report_names_target"),
                    "ordering_holds": acc(sub, "lens_correct") >= acc(sub, "model_correct"),
                }
        a_sub = [r for r in rows if r["elicitation"] == ANCHOR_CELL[0]
                 and r["strength"] == ANCHOR_CELL[1]]
        anchor = acc(a_sub, "reader_correct")
        summary = {
            "model": MODEL, "layer": LAYER, "chance": CHANCE, "smoke": args.smoke,
            "n_carriers": len(carriers), "n_rows": len(rows),
            "anchor": {
                "cell": f"{ANCHOR_CELL[0]}@{ANCHOR_CELL[1]}", "measured": anchor,
                "notes_21_expected": ANCHOR_EXPECTED,
                "within_tolerance": abs(anchor - ANCHOR_EXPECTED) <= ANCHOR_TOLERANCE
                if anchor == anchor else False,
            },
            "cells": cells,
            "leaks": sum(bool(r["report_leaks_marker"]) for r in rows),
            "elapsed_s": round(time.time() - started, 1),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2), flush=True)
        print(json.dumps(cells, indent=2), flush=True)
    finally:
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/ladder_powered_v1_raw.jsonl"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
