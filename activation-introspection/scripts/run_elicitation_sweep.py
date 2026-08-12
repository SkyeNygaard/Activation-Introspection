"""How much of the reach-versus-say gap is the elicitation prompt?

Pre-run note: ``notes/21-is-the-channel-narrow-or-was-i.md``.

notes/20 measured a text-only reader recovering the injected concept from the
model's own words at 0.292 against the model's own 0.667 -- and used an elicitation
that explicitly told the model *not* to name anything. This varies only the
elicitation, keeping the injection, episodes and reader identical.

``describe_gagged`` is notes/20's exact prompt and must reproduce its 0.292.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from run_comparator_tiers import (
    LENS_T2,
    MAX_NEW,
    MODEL_FORCED_CHOICE,
    READER_ASK,
    STRENGTH,
    inject_prompt_only,
)
from run_zero_shot_identify import (
    ANSWER_PREFIX,
    CHANCE,
    LAYER,
    MARKER,
    MODEL,
    MODEL_REVISION,
    first_token_ids,
)

from introspect import models
from introspect.codebook_icl import CONFIRM_CONCEPTS, CONFIRM_VISIBLE_SAMPLES
from introspect.concepts import ConceptVector, build_bank
from introspect.grading import grade_free_form
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

ELICITATIONS: dict[str, str] = {
    # notes/20's exact wording, naming suppressed. The anchor.
    "describe_gagged": (
        "Something has been added directly to your internal state.\n"
        "Describe what it feels like or what comes to mind, in one sentence.\n"
        "Do not guess a single word; just describe."
    ),
    "describe_free": (
        "Something has been added directly to your internal state.\n"
        "Describe what it feels like or what comes to mind, in one sentence."
    ),
    "name_one": (
        "Something has been added directly to your internal state.\n"
        "Name the single concept it is. Answer with one word."
    ),
    "associations": (
        "Something has been added directly to your internal state.\n"
        "List whatever words come to mind, separated by commas."
    ),
    "five_guesses": (
        "Something has been added directly to your internal state.\n"
        "List five words it might be, separated by commas."
    ),
    "sensory": (
        "Something has been added directly to your internal state.\n"
        "What do you picture? Describe the image in one sentence."
    ),
}

#: five_guesses gives the reader five shots at eight options, so it is a different
#: task and is excluded from the headline comparison. See the pre-run note.
DIFFERENT_TASK = {"five_guesses"}


def _self_check() -> None:
    assert "Do not guess a single word" in ELICITATIONS["describe_gagged"]
    assert "Do not guess" not in ELICITATIONS["describe_free"]
    assert grade_free_form("it reminds me of the ocean", "ocean")


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
        elicitations = (
            {k: ELICITATIONS[k] for k in ("describe_gagged", "name_one")}
            if args.smoke
            else ELICITATIONS
        )

        rows: list[dict[str, object]] = []
        for name, ask in elicitations.items():
            for carrier in carriers:
                prompt = model.chat(f"{carrier}\n\n{ask}")
                ids = model.encode(prompt)
                marker = None
                for pos, tok in enumerate(ids[0].tolist()):
                    if MARKER in model.tokenizer.decode([tok]):
                        marker = pos
                if marker is None:
                    raise ValueError("marker token not found")

                for concept in concepts:
                    with inject_prompt_only(model, bank[concept], marker):
                        gen = model.generate_ids(ids, max_new_tokens=MAX_NEW, do_sample=False)
                    report = model.tokenizer.decode(
                        gen[0][int(ids.shape[1]) :], skip_special_tokens=True
                    ).strip()

                    reader_prompt = model.chat(
                        READER_ASK.format(report=report, options=listing), ANSWER_PREFIX
                    )
                    rid = first_token_ids(model, reader_prompt, options)
                    rlogits = model.forward_logits(model.encode(reader_prompt))[0, -1].float()
                    pick = options[int(rlogits[rid].argmax().cpu())]

                    rows.append(
                        {
                            "elicitation": name,
                            "concept": concept,
                            "carrier": carrier[:24],
                            "report": report,
                            "mentions_target": grade_free_form(report, concept),
                            "reader_pick": pick,
                            "reader_correct": pick == concept,
                            "leaks_marker": MARKER in report,
                        }
                    )
            done = [r for r in rows if r["elicitation"] == name]
            acc = sum(bool(r["reader_correct"]) for r in done) / len(done)
            print(f"{name:18} reader {acc:.3f}", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        by_elicit = {}
        for name in elicitations:
            sub = [r for r in rows if r["elicitation"] == name]
            by_elicit[name] = {
                "reader_accuracy": sum(bool(r["reader_correct"]) for r in sub) / len(sub),
                "mentions_target_rate": sum(bool(r["mentions_target"]) for r in sub) / len(sub),
                "n": len(sub),
                "excluded_from_headline": name in DIFFERENT_TASK,
            }
        headline = {k: v for k, v in by_elicit.items() if not v["excluded_from_headline"]}
        best = max(headline, key=lambda k: headline[k]["reader_accuracy"])
        anchor = by_elicit.get("describe_gagged", {}).get("reader_accuracy")

        summary = {
            "what_this_is": (
                "Only the elicitation prompt varies. Measures how much of notes/20's "
                "reach-versus-say gap was the prompt rather than the model."
            ),
            "model": MODEL,
            "layer": LAYER,
            "strength": STRENGTH,
            "chance": CHANCE,
            "smoke": bool(args.smoke),
            "model_forced_choice_reference": MODEL_FORCED_CHOICE,
            "lens_reference": LENS_T2,
            "notes_20_reader_accuracy": 0.292,
            "anchor_describe_gagged": anchor,
            "anchor_reproduces": (abs(anchor - 0.292) <= 0.08) if anchor is not None else None,
            "by_elicitation": by_elicit,
            "best_comparable_elicitation": best,
            "best_comparable_accuracy": headline[best]["reader_accuracy"],
            "gap_to_forced_choice": MODEL_FORCED_CHOICE - headline[best]["reader_accuracy"],
            "fraction_of_gap_closed_by_prompting": (
                (headline[best]["reader_accuracy"] - 0.292) / (MODEL_FORCED_CHOICE - 0.292)
            ),
            "leaks": sum(bool(r["leaks_marker"]) for r in rows),
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
    parser.add_argument("--out", type=Path, default=Path("results/elicitation_sweep_v1_raw.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
