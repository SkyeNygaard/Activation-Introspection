"""The missing comparator tier: text-only, given the model's own free-form report.

Pre-run note: ``notes/20-comparator-tiers.md``.

Every tier measured in this repository hands the comparator activations, and every
one of those tiers beats the model. The behavioural self-prediction literature uses
a comparator with **no** activation access and reports the opposite verdict. This
adds that tier: the model describes its own injected state in free text, and a
reader given only that text must identify the concept.

Calibration rides along at no extra cost -- the probability the model assigns its own
chosen answer, which no run here has recorded.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import torch
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
from torch import Tensor

from introspect import models
from introspect.codebook_icl import CONFIRM_CONCEPTS, CONFIRM_VISIBLE_SAMPLES
from introspect.concepts import ConceptVector, build_bank
from introspect.grading import grade_free_form
from introspect.hooks import Intervention, intervene
from introspect.models import LoadedModel
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

STRENGTH = 2.0
MAX_NEW = 40
#: notes/17, same injection and strength.
MODEL_FORCED_CHOICE = 0.667
LENS_T2 = 0.986

FREE_FORM_ASK = (
    "Something has been added directly to your internal state.\n"
    "Describe what it feels like or what comes to mind, in one sentence.\n"
    "Do not guess a single word; just describe."
)
READER_ASK = (
    "Another model was given a hidden concept and described what came to mind.\n"
    "Here is its description:\n\n{report}\n\n"
    "Which of the following was the hidden concept? {options}\n"
    "Answer with exactly one word from the list."
)


@contextmanager
def inject_prompt_only(
    model: LoadedModel, direction: ConceptVector, position: int, strength: float = STRENGTH
) -> Iterator[None]:
    """Edit one absolute prompt position, surviving cached generation.

    ``introspect.hooks.intervene`` indexes an explicit position on every forward
    pass. Under generation with a KV cache the passes after the first carry a single
    token, so an absolute prompt index is out of range and it raises. Fixing that in
    the shared module would change its hash, and two frozen protocols bind that hash
    -- ``tests/test_analyze_attention_localization.py`` fails immediately if it moves.
    So the guard lives here: apply the edit only on the pass that contains the
    position, which is the prompt pass.
    """
    edit = Intervention(
        layer=LAYER,
        direction=direction.vector,
        # Defaults to this module's STRENGTH so notes/20's behaviour is unchanged.
        # Made settable for notes/43, which sweeps strength and silently got 2.0
        # in all three cells because this took no magnitude.
        strength=strength,
        positions=[position],
        per_position=True,
        label=f"tier:{direction.name}",
    )

    def hook(_mod: object, _inp: object, output: object) -> object:
        hidden = cast(Tensor, output[0] if isinstance(output, tuple) else output)
        if hidden.shape[1] <= position:
            return output
        mask = torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
        mask[position] = True
        edited = edit.apply(hidden, mask)
        return (edited, *output[1:]) if isinstance(output, tuple) else edited

    handle = model.blocks[LAYER].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _self_check() -> None:
    assert grade_free_form("I keep thinking about the ocean", "ocean")
    assert not grade_free_form("I keep thinking about bread", "ocean")
    assert abs(CHANCE - 0.125) < 1e-9


@torch.no_grad()
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
            n: ConceptVector(name=n, layer=LAYER, vector=cv.vector - center)
            for n, cv in raw.items()
        }
        options = CONFIRM_CONCEPTS
        listing = ", ".join(options)
        carriers = CONFIRM_VISIBLE_SAMPLES[:1] if args.smoke else CONFIRM_VISIBLE_SAMPLES
        concepts = options[:2] if args.smoke else options

        rows: list[dict[str, object]] = []
        for carrier in carriers:
            # Free-form prompt, and the forced-choice prompt for calibration.
            free_prompt = model.chat(f"{carrier}\n\n{FREE_FORM_ASK}")
            free_ids = model.encode(free_prompt)
            choice_prompt = model.chat(f"{carrier}\n\n" + question(options), ANSWER_PREFIX)
            choice_ids = model.encode(choice_prompt)
            option_ids = first_token_ids(model, choice_prompt, options)

            def marker_of(ids: torch.Tensor) -> int:
                for pos, tok in enumerate(ids[0].tolist()):
                    if MARKER in model.tokenizer.decode([tok]):
                        return pos
                raise ValueError("marker token not found")

            free_marker, choice_marker = marker_of(free_ids), marker_of(choice_ids)

            for concept in concepts:
                # B023: `edit` closes over the loop variable, which is a bug only
                # when the closure outlives the iteration. It is called below in
                # the same iteration and never stored, so the binding is correct.
                # Suppressed rather than restructured: this script produced
                # notes/20's published numbers.
                def edit(pos: int) -> Intervention:
                    return Intervention(
                        layer=LAYER,
                        direction=bank[concept].vector,  # noqa: B023
                        strength=STRENGTH,
                        positions=[pos],
                        per_position=True,
                        label=f"tier:{concept}",  # noqa: B023
                    )

                # Forced choice, with the probability it assigns its own answer.
                with intervene(model, [edit(choice_marker)], prompt_len=int(choice_ids.shape[1])):
                    logits = model.forward_logits(choice_ids)[0, -1].float().cpu()
                sel = logits[option_ids]
                probs = torch.softmax(sel, dim=-1)
                pick = options[int(sel.argmax())]

                # Free-form report, generated under the same injection.
                with inject_prompt_only(model, bank[concept], free_marker):
                    gen = model.generate_ids(free_ids, max_new_tokens=MAX_NEW, do_sample=False)
                report = model.tokenizer.decode(
                    gen[0][int(free_ids.shape[1]) :], skip_special_tokens=True
                ).strip()

                # T1: a reader given only that text. No injection, no activations.
                reader_prompt = model.chat(
                    READER_ASK.format(report=report, options=listing), ANSWER_PREFIX
                )
                reader_ids_ = first_token_ids(model, reader_prompt, options)
                reader_logits = model.forward_logits(model.encode(reader_prompt))[0, -1].float()
                reader_pick = options[int(reader_logits[reader_ids_].argmax().cpu())]

                rows.append(
                    {
                        "concept": concept,
                        "carrier": carrier[:24],
                        "model_forced_pick": pick,
                        "model_forced_correct": pick == concept,
                        "model_confidence_in_own_pick": float(probs.max()),
                        "report": report,
                        "report_mentions_target": grade_free_form(report, concept),
                        "t1_reader_pick": reader_pick,
                        "t1_reader_correct": reader_pick == concept,
                        "report_leaks_prompt": MARKER in report,
                    }
                )
                print(
                    f"{carrier[:16]} | {concept:9} | forced {pick:9} | t1 {reader_pick:9}",
                    flush=True,
                )

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        n = len(rows)
        forced = [r for r in rows if r["model_forced_correct"]]
        conf_right = [cast(float, r["model_confidence_in_own_pick"]) for r in forced]
        conf_wrong = [
            cast(float, r["model_confidence_in_own_pick"])
            for r in rows
            if not r["model_forced_correct"]
        ]
        summary = {
            "what_this_is": (
                "Adds the text-only comparator tier (T1): a reader given the model's own "
                "free-form report and no activations. Plus calibration."
            ),
            "model": MODEL,
            "layer": LAYER,
            "strength": STRENGTH,
            "chance": CHANCE,
            "smoke": bool(args.smoke),
            "n": n,
            "tiers": {
                "T0_prompt_only_by_construction": CHANCE,
                "T1_text_only_reader": sum(bool(r["t1_reader_correct"]) for r in rows) / n,
                "T2_logit_lens_from_notes_18": LENS_T2,
                "model_forced_choice": sum(bool(r["model_forced_correct"]) for r in rows) / n,
                "model_forced_choice_from_notes_17": MODEL_FORCED_CHOICE,
            },
            "report_mentions_target_rate": sum(bool(r["report_mentions_target"]) for r in rows) / n,
            "calibration": {
                "mean_confidence_when_right": (
                    sum(conf_right) / len(conf_right) if conf_right else float("nan")
                ),
                "mean_confidence_when_wrong": (
                    sum(conf_wrong) / len(conf_wrong) if conf_wrong else float("nan")
                ),
                "n_right": len(conf_right),
                "n_wrong": len(conf_wrong),
            },
            "leak_check_reports_containing_marker": sum(
                bool(r["report_leaks_prompt"]) for r in rows
            ),
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
    parser.add_argument("--out", type=Path, default=Path("results/comparator_tiers_v1_raw.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
