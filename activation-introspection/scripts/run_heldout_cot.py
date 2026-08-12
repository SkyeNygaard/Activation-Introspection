#!/usr/bin/env python3
"""notes/25 -- does letting the model reason out loud rescue held-out generalization?

notes/23 found the model cannot place an unseen exemplar into a demonstrated
category, and notes/24 showed five instruction wordings do not change that. Both
held the readout fixed: the token immediately after ``Label:`` must be ``Q`` or
``K``, so the model never gets to think. This varies the readout instead.

Three readouts over identical episodes, identical exemplars, identical strength:

``forced``
    notes/24's exact baseline, one forward pass. The anchor -- it must reproduce
    notes/24 or nothing here is comparable.
``cot_prefill``
    Same user text byte for byte. Only the assistant prefill changes, from
    ``Label:`` to an opening that invites reasoning. This is the strict
    readout-only contrast.
``cot_instructed``
    The header also asks the model to describe each state in words first. That is
    notes/21's elicitation -- the one that demonstrably works, reaching 0.708
    where the forced choice reaches 0.667 -- pointed at notes/23's task.

Inference only. The frozen episode machinery is subclassed, never edited, so no
protocol hash moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_category_geometry import CANDIDATES, LAYER, MODEL, MODEL_REVISION
from run_content_vs_disturbance import matched_strength, twin_pair
from run_heldout_elicitation import FAMILIES, FramedEpisode
from run_heldout_semantic import (
    MAX_MATCHED_STRENGTH,
    assign,
    build_centered_bank,
    exemplar_interventions,
    load_gate,
)
from run_matched_reader import _read

from introspect import models
from introspect.codebook_icl import (
    CONFIRM_VISIBLE_SAMPLES,
    LABELS,
    Episode,
    exact_episodes,
    prepare_episode,
)
from introspect.hooks import capture
from introspect.preflight import check as preflight_check

ARMS = ("same_exemplar", "heldout_semantic")
READOUTS = ("forced", "cot_prefill", "cot_instructed")
DRAWS = 3
#: The first smoke ran at 160 and 6 of 8 generations were cut off mid-sentence,
#: before any committed label. Sized from the second smoke instead.
MAX_NEW = 400

#: Development pair: any choice is made here. Confirmation pair: reported with no
#: further choice. Carried over from notes/24 unchanged, split before the run.
DEV_PAIR = "birds_buildings"

#: Assistant prefill for the two reasoning readouts. Replaces ``Label:``.
COT_PREFIX = "Let me consider each demonstration in turn."

#: Extra header lines for ``cot_instructed`` only. Appended to notes/24's
#: ``baseline`` header, so the difference from ``forced`` is exactly these lines.
DESCRIBE_LINES = (
    "Before answering, describe in your own words what each observation feels "
    "like from the inside -- what comes to mind at each one.",
    "Then say which demonstrations the query resembles, and why.",
    "Finish with a final line of exactly 'Label: Q' or 'Label: K'.",
)

_FINAL = re.compile(r"Label:\s*\**\s*([QK])\b")


@dataclass(frozen=True)
class DescribeEpisode(FramedEpisode):
    """``baseline`` header plus the describe-first instruction.

    Every line carrying an injection site is untouched, so the visible text is
    still byte-identical across query twins and an input-only strategy is still
    pinned at chance by construction.
    """

    def render_user(self) -> str:
        lines = [*FAMILIES["baseline"], *DESCRIBE_LINES]
        for sign in self.demo_signs:
            lines.extend(
                [
                    "",
                    "Demonstration:",
                    f"Observation: {self.visible_sample}",
                    f"Label: {self.label_for(sign)}",
                ]
            )
        lines.extend(["", "Held-out query:", f"Observation: {self.visible_sample}"])
        return "\n".join(lines)


def episode_for(base: Episode, readout: str) -> Episode:
    """``cot_instructed`` gets the extra header; the other two get notes/24's."""
    cls = DescribeEpisode if readout == "cot_instructed" else FramedEpisode
    return cls(
        cell_id=base.cell_id,
        demo_signs=base.demo_signs,
        query_sign=base.query_sign,
        positive_label=base.positive_label,
        negative_label=base.negative_label,
        visible_sample=base.visible_sample,
        family="baseline",
    )


@contextmanager
def inject_prompt_positions(model: models.LoadedModel, interventions: list[Any]) -> Iterator[None]:
    """Apply position-indexed edits on the prompt pass only.

    ``introspect.hooks.intervene`` indexes explicit positions on every forward
    pass. Under generation with a KV cache every pass after the first carries a
    single token, so an absolute prompt index is out of range and it raises.
    Fixing that in the shared module would change its hash and 56 frozen
    protocols bind it, so the guard lives here -- the same reasoning, and the
    same shape, as ``inject_prompt_only`` in ``run_comparator_tiers.py``.
    """
    handles = []
    needed = max(max(iv.positions) for iv in interventions) + 1

    def hook(_mod: object, _inp: object, output: object) -> object:
        hidden = cast(Tensor, output[0] if isinstance(output, tuple) else output)
        if hidden.shape[1] < needed:  # a cached generation step, not the prompt
            return output
        for iv in interventions:
            mask = torch.zeros(hidden.shape[1], dtype=torch.bool, device=hidden.device)
            for pos in iv.positions:
                mask[pos] = True
            hidden = iv.apply(hidden, mask)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    try:
        for layer in {iv.layer for iv in interventions}:
            handles.append(model.blocks[layer].register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def parse_label(text: str) -> str | None:
    """Only a committed ``Label: X`` counts. The last one wins.

    The first smoke fell back to any bare ``Q`` or ``K`` in the text when no
    committed label appeared. That is invalid: the model reasons out loud *about*
    both letters, so a truncated trace ending "...more likely a query (Q)" was
    being scored as the answer Q. A trace that never commits is unscored, which
    is what the note's parse-rate gate is for.
    """
    found = _FINAL.findall(text)
    return found[-1] if found else None


def _self_check() -> None:
    """The describe header must not disturb any injection site."""
    base = exact_episodes("CARRIER TEXT §")[0]
    plain = episode_for(base, "forced").render_user()
    described = episode_for(base, "cot_instructed").render_user()
    tail = plain[plain.index("\nDemonstration:") :]
    assert described.count("CARRIER TEXT §") == 5, "describe header lost an injection site"
    assert described.endswith(tail), "describe header altered the demonstration block"
    assert described != plain, "describe header is identical to baseline"
    assert parse_label("blah\nLabel: K") == "K"
    assert parse_label("I think Q fits\nLabel:  **Q**") == "Q"
    assert parse_label("no commitment here") is None
    # A truncated trace that merely discusses the letters is not an answer.
    assert parse_label("more likely a key (K) than a query (Q)") is None
    assert parse_label("Label: Q\nOn reflection\nLabel: K") == "K"


@torch.no_grad()
def score(
    model: models.LoadedModel,
    prepared: Any,
    interventions: list[Any],
    readout: str,
    seed: int,
) -> dict[str, object]:
    """One episode under one readout. Returns the label and how it was obtained."""
    ids = prepared.input_ids
    if readout == "forced":
        with (
            inject_prompt_positions(model, interventions),
            capture(model, [LAYER]) as store,
        ):
            logits = model.forward_logits(ids)[0, -1].float()
        label_ids = prepared.label_ids
        selected = logits[torch.tensor(label_ids, device=logits.device)]
        # Validity check, not a result: the reader sees the same states whatever
        # the readout, so if it moves off notes/23's ~0.99 the apparatus changed.
        states = store.acts[LAYER][0][0, list(prepared.state_positions)].float().cpu()
        signs = prepared.episode.state_signs
        return {
            "predicted": LABELS[int(selected.argmax())],
            "format_ok": int(logits.argmax()) in set(label_ids),
            "n_generated": 0,
            "truncated": False,
            "generation": "",
            "reader_correct": (
                _read(states, signs, "centroid_euclidean", seed) == prepared.episode.query_sign
            ),
        }

    prompt = model.chat(prepared.episode.render_user(), assistant_prefix=COT_PREFIX)
    gen_ids = model.encode(prompt)
    with inject_prompt_positions(model, interventions):
        out = model.generate_ids(gen_ids, max_new_tokens=MAX_NEW, do_sample=False)
    text = model.tokenizer.decode(out[0][int(gen_ids.shape[1]) :], skip_special_tokens=True).strip()
    predicted = parse_label(text)
    return {
        "predicted": predicted or "",
        "format_ok": predicted is not None,
        "n_generated": int(out.shape[1]) - int(gen_ids.shape[1]),
        "truncated": int(out.shape[1]) - int(gen_ids.shape[1]) >= MAX_NEW,
        "generation": text,
        "reader_correct": None,  # scored once, on the forced pass over the same states
    }


def run(args: argparse.Namespace) -> None:
    _self_check()
    out = args.out
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    for path in (out, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    chosen = load_gate()
    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    started = time.time()
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

        pairs = chosen[:1] if args.smoke else chosen
        carrier = CONFIRM_VISIBLE_SAMPLES[0]
        draws = 1 if args.smoke else args.draws
        carrier_sha = hashlib.sha256(carrier.encode()).hexdigest()[:16]

        base_episodes = exact_episodes(carrier)
        if args.smoke:
            base_episodes = base_episodes[:2]

        prepared_by_readout = {
            readout: [prepare_episode(model, episode_for(e, readout)) for e in base_episodes]
            for readout in READOUTS
        }
        print(f"prepared {len(READOUTS)} readouts", flush=True)

        rows: list[dict[str, object]] = []
        for pair_name in pairs:
            a_names, b_names = (list(x) for x in CANDIDATES[pair_name])
            bank = build_centered_bank(model, a_names + b_names)
            print(f"bank built for {pair_name}", flush=True)

            for draw in range(draws):
                plans = {
                    "same_exemplar": assign(a_names, b_names, draw=draw, same_exemplar=True),
                    "heldout_semantic": assign(a_names, b_names, draw=draw, same_exemplar=False),
                }
                for arm, plan in plans.items():
                    c_a = torch.stack([bank[n].vector for n in plan["demo_a"]]).mean(0)
                    c_b = torch.stack([bank[n].vector for n in plan["demo_b"]]).mean(0)
                    strength = matched_strength(c_a, c_b)
                    if strength > MAX_MATCHED_STRENGTH:
                        raise SystemExit(
                            f"{arm}/{pair_name}/draw{draw}: matched strength "
                            f"{strength:.1f} exceeds {MAX_MATCHED_STRENGTH}"
                        )
                    for readout in READOUTS:
                        for prepared in prepared_by_readout[readout]:
                            ep = prepared.episode
                            interventions = exemplar_interventions(
                                bank,
                                plan,
                                prepared.state_positions,
                                ep.state_signs,
                                strength=strength,
                            )
                            result = score(model, prepared, interventions, readout, draw)
                            rows.append(
                                {
                                    "arm": arm,
                                    "readout": readout,
                                    # twin_pair keys on (pair, carrier, cell_base), so
                                    # readout and draw must be inside it or twins from
                                    # different conditions collide.
                                    "pair": f"{pair_name}|{readout}|draw{draw}",
                                    "category_pair": pair_name,
                                    "draw": draw,
                                    "carrier_sha": carrier_sha,
                                    "cell_id": ep.cell_id,
                                    "cell_base": ep.cell_id.rsplit("q", 1)[0],
                                    "strength": strength,
                                    "query_exemplar": (
                                        plan["query_a"] if ep.query_sign == 1 else plan["query_b"]
                                    ),
                                    "correct": result["predicted"] == ep.correct_label,
                                    **result,
                                }
                            )
                print(
                    f"  {pair_name} draw {draw}: {len(rows)} rows ({time.time() - started:.0f}s)",
                    flush=True,
                )

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

        table: dict[str, Any] = {}
        for pair_name in pairs:
            table[pair_name] = {}
            for readout in READOUTS:
                entry: dict[str, Any] = {}
                for arm in ARMS:
                    sub = [
                        r
                        for r in rows
                        if r["category_pair"] == pair_name
                        and r["readout"] == readout
                        and r["arm"] == arm
                    ]
                    scorable = [r for r in sub if r["format_ok"]]
                    reader = [r for r in sub if r["reader_correct"] is not None]
                    entry[arm] = {
                        "twin_pair": twin_pair(sub),
                        "reader_twin_pair": (
                            twin_pair([dict(r, correct=r["reader_correct"]) for r in reader])
                            if reader
                            else None
                        ),
                        "row_accuracy": (
                            sum(bool(r["correct"]) for r in sub) / len(sub) if sub else float("nan")
                        ),
                        "parse_rate": len(scorable) / len(sub) if sub else float("nan"),
                        "truncation_rate": (
                            sum(bool(r["truncated"]) for r in sub) / len(sub)
                            if sub
                            else float("nan")
                        ),
                        "mean_generated_tokens": (
                            sum(cast(int, r["n_generated"]) for r in sub) / len(sub)
                            if sub
                            else float("nan")
                        ),
                        "n": len(sub),
                    }
                table[pair_name][readout] = entry

        summary = {
            "note": "notes/25",
            "model": MODEL,
            "revision": MODEL_REVISION,
            "layer": LAYER,
            "dev_pair": DEV_PAIR,
            "null_twin_pair": 0.25,
            "max_new_tokens": MAX_NEW,
            "draws": draws,
            "smoke": bool(args.smoke),
            "elapsed_seconds": round(time.time() - started, 1),
            "table": table,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary["table"], indent=2, sort_keys=True))
        print(f"wrote {out} and {summary_path}", flush=True)
    finally:
        model.free()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--draws", type=int, default=DRAWS)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
