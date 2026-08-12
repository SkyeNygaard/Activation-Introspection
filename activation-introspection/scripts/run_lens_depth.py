"""Is there any read depth where the logit lens falls below the model?

Pre-run note: ``notes/18-where-the-lens-fails.md``, written before this ran.

notes/17 argued the cost criterion is unsatisfiable wherever a state is linearly
decodable, and named the check nobody has run: find the depths where a lens
*cannot* read. This sweeps every block from the injection site to the last, at the
injection position and at the answer position, against the model's fixed 0.597.

Injection, prompts and options are imported from run_zero_shot_identify so the two
runs are the same experiment read at different depths.
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

STRENGTH = 2.0
#: notes/17, same injection, model reading its own state at this strength.
MODEL_ACCURACY = 0.667


def lens_all(model: models.LoadedModel, state: Tensor) -> Tensor:
    inner = cast(Any, model.model)
    hidden = state.to(model.device, model.dtype).unsqueeze(0).unsqueeze(0)
    return cast(Tensor, inner.lm_head(inner.model.norm(hidden)))[0, 0].float().cpu()


def _self_check() -> None:
    assert abs(CHANCE - 0.125) < 1e-9
    assert MODEL_ACCURACY > CHANCE, "the model's own number must beat chance to compare against"


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
        n_blocks = len(model.blocks)
        depths = list(range(LAYER, n_blocks))
        print(f"{n_blocks} blocks; sweeping {depths[0]}..{depths[-1]}", flush=True)

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

        rows: list[dict[str, object]] = []
        for carrier in carriers:
            prompt = model.chat(f"{carrier}\n\n" + question(options), ANSWER_PREFIX)
            ids = model.encode(prompt)
            option_ids = first_token_ids(model, prompt, options)
            marker_pos = None
            for pos, tok in enumerate(ids[0].tolist()):
                if MARKER in model.tokenizer.decode([tok]):
                    marker_pos = pos
            if marker_pos is None:
                raise ValueError("marker token not found")
            final_pos = int(ids.shape[1]) - 1

            for concept in concepts:
                edit = Intervention(
                    layer=LAYER,
                    direction=bank[concept].vector,
                    strength=STRENGTH,
                    positions=[marker_pos],
                    per_position=True,
                    label=f"lensdepth:{concept}",
                )
                with (
                    intervene(model, [edit], prompt_len=int(ids.shape[1])),
                    capture(model, depths) as store,
                ):
                    model.forward_logits(ids)
                for depth in depths:
                    acts = store.acts[depth][0][0]
                    for site, pos in (("marker", marker_pos), ("final", final_pos)):
                        scores = lens_all(model, acts[pos].clone())[option_ids]
                        pick = options[int(scores.argmax())]
                        rows.append(
                            {
                                "depth": depth,
                                "site": site,
                                "concept": concept,
                                "carrier": carrier[:24],
                                "predicted": pick,
                                "correct": pick == concept,
                            }
                        )
                print(f"{carrier[:18]}: {concept}", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        by_depth: dict[str, dict[str, float]] = {}
        for depth in depths:
            entry = {}
            for site in ("marker", "final"):
                sub = [r for r in rows if r["depth"] == depth and r["site"] == site]
                entry[site] = sum(bool(r["correct"]) for r in sub) / len(sub)
            by_depth[str(depth)] = entry

        below = [
            {"depth": d, "site": s, "lens": v[s]}
            for d, v in by_depth.items()
            for s in ("marker", "final")
            if v[s] < MODEL_ACCURACY
        ]
        best_site = {
            site: max(by_depth.values(), key=lambda v: v[site])[site] for site in ("marker", "final")
        }
        # The criterion asks whether SOME cheap reader beats the model, so the
        # third party is allowed to pick its best depth and site.
        third_party_best = max(best_site.values())

        summary = {
            "what_this_is": (
                "Logit-lens identification accuracy at every block from the injection "
                "site to the last, against the model's own 0.667 from notes/17."
            ),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "injection_layer": LAYER,
            "strength": STRENGTH,
            "chance": CHANCE,
            "model_accuracy_from_notes_17": MODEL_ACCURACY,
            "smoke": bool(args.smoke),
            "n_blocks": n_blocks,
            "by_depth": by_depth,
            "best_per_site": best_site,
            "third_party_best_over_all_depths": third_party_best,
            "depths_where_lens_below_model": below,
            "n_depths_below_model": len(below),
            "verdict": (
                "criterion_unsatisfiable_at_every_depth"
                if third_party_best >= MODEL_ACCURACY
                else "some_depth_favours_the_model"
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
    parser.add_argument("--out", type=Path, default=Path("results/lens_depth_v1_raw.jsonl"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
