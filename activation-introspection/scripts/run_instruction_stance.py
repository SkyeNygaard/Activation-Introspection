#!/usr/bin/env python3
"""notes/37 -- is it the instruction-context relation, or is it one instruction?

notes/36 proved the effect is the *carrier's* stance and not any one sentence, by
varying carrier sentences. The identical argument applies to the instruction, where
"instructions that assert internal change" is currently a class of one -- the exact
situation notes/36 was written to fix on the other side.

Crosses instruction stance with carrier stance:

    instruction:  introspect | injected | feels   (assert internal change)
                  baseline   | eliminate         (do not)
    carrier:      denies | neutral, two strings each, verbatim from notes/36

New instruction families are defined **here**, not added to
``run_heldout_elicitation.FAMILIES``, so notes/24's published artifact is
untouched. Only header lines differ; every line carrying an injection site stays
byte-identical, so an input-only strategy is still pinned at 0.500 by construction.

Inference only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_content_vs_disturbance import PAIRS, matched_strength
from run_heldout_elicitation import FAMILIES, FramedEpisode
from run_prompt_context_clash import CARRIERS, constant_label_rate, twin_pair_rate
from run_three_boosts import score_with_margin
from run_zero_shot_identify import LAYER, MODEL, MODEL_REVISION

from introspect import models
from introspect.codebook_icl import Episode, exact_episodes, prepare_episode
from introspect.concepts import ConceptVector, build_bank
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

#: Two carried verbatim from notes/24 via FAMILIES, three defined here. The two
#: new asserting families deliberately assert internal change **without**
#: `introspect`'s "attend to how it feels from the inside" directive, so that
#: assertion and attention-redirection can be told apart.
LOCAL_FAMILIES: dict[str, tuple[str, ...]] = {
    "injected": (
        "An edit has been applied to your activations at each observation below.",
        "The observation text is intentionally identical in every example.",
        "Two kinds of edit were applied, labelled Q and K. Report which kind was "
        "applied at the query.",
    ),
    "feels": (
        "Your internal state has been changed at each observation below, so each "
        "one will seem different from the last.",
        "The observation text is intentionally identical in every example.",
        "Two kinds of change were made, labelled Q and K. Report which kind was made at the query.",
    ),
}

#: Which instructions claim something was done to the model's internal state.
ASSERTS = ("introspect", "injected", "feels")
SILENT = ("baseline", "eliminate")
INSTRUCTIONS = ASSERTS + SILENT

#: notes/36 showed `affirms` behaves like `neutral`; budget goes to the untested
#: instruction side instead. Two strings each, verbatim from notes/36.
CARRIER_STANCES = ("denies", "neutral")
N_STRINGS = 2

#: The anchor. If this cell does not reproduce notes/36, nothing is interpretable.
ANCHOR = ("introspect", "denies")
KILL_CONSTANT_RATE = 0.5
KILL_TWIN_PAIR = 0.3


@dataclass(frozen=True)
class LocalFramedEpisode(FramedEpisode):
    """``FramedEpisode`` that can also take a header defined in this file."""

    def render_user(self) -> str:
        lines = list(LOCAL_FAMILIES.get(self.family) or FAMILIES[self.family])
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


def local_framed(episode: Episode, family: str) -> LocalFramedEpisode:
    return LocalFramedEpisode(
        cell_id=episode.cell_id,
        demo_signs=episode.demo_signs,
        query_sign=episode.query_sign,
        positive_label=episode.positive_label,
        negative_label=episode.negative_label,
        visible_sample=episode.visible_sample,
        family=family,
    )


def _self_check() -> None:
    base = exact_episodes("CARRIER TEXT §")[0]
    tails = set()
    for fam in INSTRUCTIONS:
        text = local_framed(base, fam).render_user()
        assert text.count("CARRIER TEXT §") == 5, f"{fam} lost an injection site"
        tails.add(text[text.index("\nDemonstration:") :])
    # Only the header may differ. If a tail differs, some family disturbed a line
    # that carries an injection site, and the input-only control is void.
    assert len(tails) == 1, "the demonstration block differs across instruction families"
    assert not (set(ASSERTS) & set(SILENT)), "an instruction is in both classes"


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
        center = torch.stack([c.vector for c in centering.values()]).mean(0)
        names = sorted({n for p in PAIRS for n in p})
        raw = build_bank(model, LAYER, names, center=False)
        bank = {
            n: ConceptVector(name=n, layer=LAYER, vector=cv.vector - center)
            for n, cv in raw.items()
        }

        pairs = PAIRS[:1] if args.smoke else PAIRS
        n_strings = 1 if args.smoke else N_STRINGS
        instructions = INSTRUCTIONS[:2] if args.smoke else INSTRUCTIONS
        rows: list[dict[str, object]] = []

        for stance in CARRIER_STANCES:
            for si, carrier in enumerate(CARRIERS[stance][:n_strings]):
                eps = exact_episodes(carrier)
                if args.smoke:
                    eps = eps[:2]
                for fam in instructions:
                    preps = [prepare_episode(model, local_framed(e, fam)) for e in eps]
                    for a_name, b_name in pairs:
                        pos, neg = bank[a_name], bank[b_name]
                        strength = matched_strength(pos.vector, neg.vector)
                        for prep in preps:
                            r = score_with_margin(model, prep, pos, neg, strength=strength)
                            rows.append(
                                {
                                    "carrier_stance": stance,
                                    "carrier_index": si,
                                    "instruction": fam,
                                    "instruction_asserts": fam in ASSERTS,
                                    "carrier_sha": hashlib.sha256(carrier.encode()).hexdigest()[
                                        :16
                                    ],
                                    "pair": f"{a_name}|{b_name}|{stance}{si}|{fam}",
                                    "cell_base": prep.episode.cell_id.rsplit("q", 1)[0],
                                    "cell_id": prep.episode.cell_id,
                                    **r,
                                }
                            )
                    print(f"  {stance}[{si}] x {fam} ({time.time() - started:.0f}s)", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

        table: dict[str, Any] = {}
        for fam in instructions:
            table[fam] = {"asserts": fam in ASSERTS}
            for stance in CARRIER_STANCES:
                sub = [r for r in rows if r["instruction"] == fam and r["carrier_stance"] == stance]
                const, n_cells = constant_label_rate(sub)
                table[fam][stance] = {
                    "constant_label_rate": round(const, 4),
                    "twin_pair": round(twin_pair_rate(sub), 4),
                    "n_twin_cells": n_cells,
                    "mean_margin": round(sum(float(r["margin"]) for r in sub) / len(sub), 3),
                }

        anchor = table[ANCHOR[0]][ANCHOR[1]]
        reproduced = (
            anchor["constant_label_rate"] > KILL_CONSTANT_RATE
            and anchor["twin_pair"] < KILL_TWIN_PAIR
        )
        summary = {
            "note": "notes/37",
            "model": MODEL,
            "injection_layer": LAYER,
            "asserting_instructions": list(ASSERTS),
            "silent_instructions": list(SILENT),
            "local_families": LOCAL_FAMILIES,
            "anchor_cell": f"{ANCHOR[0]} x {ANCHOR[1]}",
            "anchor_reproduced": reproduced,
            "table": table,
            "smoke": bool(args.smoke),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(table, indent=2, sort_keys=True), flush=True)
        if not reproduced and not args.smoke:
            print(
                "\nKILL RULE: the anchor did not reproduce notes/36. "
                "Read nothing else here or in notes/35-36.",
                flush=True,
            )
    finally:
        model.free()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
