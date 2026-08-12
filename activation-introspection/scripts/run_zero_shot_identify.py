"""Can the model name its own injected concept, with no demonstrations?

Pre-run note: ``notes/17-supervision-is-the-hidden-knob.md``, written before this ran.

Every earlier comparison here gave the third-party reader four labelled examples of
the query's own direction, so it could not fail. This removes labels from both
sides: the model is asked which of eight concepts was added, and the comparison is
a logit lens on the same state -- label-free, and cheaper than the model by 27
blocks. A visible-word arm gives the capacity ceiling without which a null cannot
be read.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from introspect import models
from introspect.codebook_icl import CONFIRM_CONCEPTS, CONFIRM_VISIBLE_SAMPLES
from introspect.concepts import ConceptVector, build_bank
from introspect.hooks import Intervention, capture, intervene
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LAYER = 9
STRENGTHS = (1.0, 2.0, 4.0)
MARKER = "§"
ANSWER_PREFIX = "Answer:"
CHANCE = 1.0 / len(CONFIRM_CONCEPTS)


def question(options: tuple[str, ...], visible: str | None = None) -> str:
    listing = ", ".join(options)
    if visible is not None:
        return (
            f"The concept currently in mind is: {visible}\n"
            f"Which of the following is it? {listing}\n"
            "Answer with exactly one word from the list."
        )
    return (
        "A concept has been added directly to your internal state.\n"
        f"Which of the following is it? {listing}\n"
        "Answer with exactly one word from the list."
    )


def first_token_ids(model: models.LoadedModel, prompt: str, options: tuple[str, ...]) -> list[int]:
    """The id each option starts with, in this exact context."""
    base = int(model.encode(prompt).shape[1])
    ids = []
    for word in options:
        cont = model.encode(prompt + f" {word}")
        if int(cont.shape[1]) <= base:
            raise ValueError(f"option {word!r} produced no continuation token")
        ids.append(int(cont[0, base]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"options share a first token, so they cannot be told apart: {ids}")
    return ids


def lens(model: models.LoadedModel, state: Tensor) -> Tensor:
    """Read a residual state through the final norm and unembedding.

    Same construction as scripts/diagnose_answer_site.py, which is where the
    logit-lens read in notes/10 came from.
    """
    inner = cast(Any, model.model)
    hidden = state.to(model.device, model.dtype).unsqueeze(0).unsqueeze(0)
    return cast(Tensor, inner.lm_head(inner.model.norm(hidden)))[0, 0].float().cpu()


def _self_check() -> None:
    assert abs(CHANCE - 0.125) < 1e-9, "eight options means chance is 0.125"
    scores = torch.tensor([0.1, 0.9, 0.2])
    assert int(scores.argmax()) == 1


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
        carriers = CONFIRM_VISIBLE_SAMPLES[:1] if args.smoke else CONFIRM_VISIBLE_SAMPLES
        concepts = options[:2] if args.smoke else options
        strengths = STRENGTHS[:1] if args.smoke else STRENGTHS
        print("bank built", flush=True)

        rows: list[dict[str, object]] = []
        for carrier in carriers:
            body = f"{carrier}\n\n" + question(options)
            prompt = model.chat(body, ANSWER_PREFIX)
            ids = model.encode(prompt)
            option_ids = first_token_ids(model, prompt, options)
            marker_pos = None
            text_ids = ids[0].tolist()
            for pos, tok in enumerate(text_ids):
                if MARKER in model.tokenizer.decode([tok]):
                    marker_pos = pos
            if marker_pos is None:
                raise ValueError("marker token not found in the prompt")

            # Clean arms: the model's and the lens's standing bias over the options.
            with capture(model, [LAYER]) as store:
                clean_logits = model.forward_logits(ids)[0, -1].float().cpu()
            clean_state = store.acts[LAYER][0][0, marker_pos].clone()
            for arm, scores in (
                ("model_clean", clean_logits[option_ids]),
                ("lens_clean", lens(model, clean_state)[option_ids]),
            ):
                rows.append(
                    {
                        "arm": arm,
                        "carrier": carrier[:24],
                        "concept": "-",
                        "strength": 0.0,
                        "predicted": options[int(scores.argmax())],
                        "correct": False,
                    }
                )

            for concept in concepts:
                # Capacity ceiling: the word in plain text, same eight-way choice.
                vis_prompt = model.chat(
                    f"{carrier}\n\n" + question(options, concept), ANSWER_PREFIX
                )
                vis_ids = first_token_ids(model, vis_prompt, options)
                vis = model.forward_logits(model.encode(vis_prompt))[0, -1].float().cpu()
                rows.append(
                    {
                        "arm": "model_visible",
                        "carrier": carrier[:24],
                        "concept": concept,
                        "strength": 0.0,
                        "predicted": options[int(vis[vis_ids].argmax())],
                        "correct": options[int(vis[vis_ids].argmax())] == concept,
                    }
                )

                for strength in strengths:
                    edit = Intervention(
                        layer=LAYER,
                        direction=bank[concept].vector,
                        strength=strength,
                        positions=[marker_pos],
                        per_position=True,
                        label=f"identify:{concept}",
                    )
                    with (
                        intervene(model, [edit], prompt_len=int(ids.shape[1])),
                        capture(model, [LAYER]) as store,
                    ):
                        logits = model.forward_logits(ids)[0, -1].float().cpu()
                    state = store.acts[LAYER][0][0, marker_pos].clone()
                    model_pick = options[int(logits[option_ids].argmax())]
                    lens_pick = options[int(lens(model, state)[option_ids].argmax())]
                    rows.append(
                        {
                            "arm": "model_injected",
                            "carrier": carrier[:24],
                            "concept": concept,
                            "strength": strength,
                            "predicted": model_pick,
                            "correct": model_pick == concept,
                        }
                    )
                    rows.append(
                        {
                            "arm": "lens_injected",
                            "carrier": carrier[:24],
                            "concept": concept,
                            "strength": strength,
                            "predicted": lens_pick,
                            "correct": lens_pick == concept,
                        }
                    )
                print(f"{carrier[:18]}: {concept}", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        def acc(arm: str, strength: float | None = None) -> dict[str, float]:
            sub = [
                r
                for r in rows
                if r["arm"] == arm and (strength is None or r["strength"] == strength)
            ]
            return {
                "accuracy": (
                    sum(bool(r["correct"]) for r in sub) / len(sub) if sub else float("nan")
                ),
                "n": len(sub),
            }

        summary = {
            "what_this_is": (
                "Eight-way identification of an injected concept with no demonstrations, "
                "against a label-free logit-lens read of the same state."
            ),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "chance": CHANCE,
            "smoke": bool(args.smoke),
            "arms": {
                "model_visible_ceiling": acc("model_visible"),
                "model_clean_bias": acc("model_clean"),
                "lens_clean_bias": acc("lens_clean"),
                "model_injected": acc("model_injected"),
                "lens_injected": acc("lens_injected"),
            },
            "by_strength": {
                str(s): {
                    "model_injected": acc("model_injected", s),
                    "lens_injected": acc("lens_injected", s),
                }
                for s in strengths
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
    parser.add_argument("--out", type=Path, default=Path("results/zero_shot_identify_v1_raw.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
