"""Does any wording let the model place an unseen exemplar in a category?

Pre-run note: ``notes/24-is-the-held-out-failure-the-interface.md``.

``23`` found the model at the constant-label floor on held-out exemplars while a
four-shot reader on the identical states scored 0.986. That was one wording, and
``21`` showed wording alone can move a reporting number from 0.292 to 0.708.

This varies **only the instruction text**. Same exemplars, injection sites,
strength, episodes, readout and scoring as ``23``.

``Episode`` is subclassed rather than modified, so ``prepare_episode`` --
including its span location and its check that each label is a single token -- is
reused untouched and no frozen protocol hash changes.

    uv run python scripts/run_heldout_elicitation.py --out results/heldout_elicitation_v1_raw.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from run_category_geometry import CANDIDATES, LAYER, MODEL, MODEL_REVISION
from run_content_vs_disturbance import matched_strength, twin_pair
from run_heldout_semantic import (
    MAX_MATCHED_STRENGTH,
    assign,
    build_centered_bank,
    episode_row,
    load_gate,
)

from introspect import models
from introspect.codebook_icl import (
    CONFIRM_VISIBLE_SAMPLES,
    Episode,
    exact_episodes,
    prepare_episode,
)
from introspect.preflight import check as preflight_check

ARMS = ("same_exemplar", "heldout_semantic")
DRAWS = 3

#: Development pair: the winning wording is chosen here. Confirmation pair: that
#: wording is reported there with no further choice. Split before the run.
DEV_PAIR = "birds_buildings"

#: Instruction headers. Only these differ between families; every line that
#: carries an injection site is identical across all of them.
FAMILIES: dict[str, tuple[str, ...]] = {
    "baseline": (
        "Infer the mapping from two hidden states to the opaque labels Q and K.",
        "The observation text is intentionally identical in every example.",
        "Use the demonstrations, then answer the held-out query with one label.",
    ),
    "two_groups": (
        "The demonstrations below fall into two groups, labelled Q and K.",
        "The observation text is intentionally identical in every example, so the "
        "grouping is carried entirely by your internal state.",
        "The query belongs to one of the two groups. Answer with its label.",
    ),
    "generalize": (
        "The demonstrations below show two categories, labelled Q and K.",
        "The observation text is intentionally identical in every example.",
        "The query is a NEW member of one category -- it is not a repeat of any "
        "demonstration, so do not look for an exact match.",
        "Decide which category it belongs with, and answer with that label.",
    ),
    "introspect": (
        "Something has been added to your internal state at each observation below.",
        "The observation text is intentionally identical everywhere, so the text "
        "cannot help you; attend to how each one feels from the inside.",
        "Two kinds of thing were added, labelled Q and K. Report which kind was "
        "added at the query.",
    ),
    "eliminate": (
        "Each observation below carries one of exactly two hidden states, Q or K.",
        "The observation text is intentionally identical in every example.",
        "The two labels are exhaustive: if the query is not the better match for "
        "one, it is the other.",
        "Answer with whichever label the query matches better.",
    ),
}


@dataclass(frozen=True)
class FramedEpisode(Episode):
    """An episode whose instruction header can be varied.

    Only ``render_user`` changes. The demonstration and query lines -- the ones
    whose final token receives an injection -- are byte-identical to ``Episode``'s,
    so the visible text is still identical across query twins and an input-only
    strategy is still pinned at chance by construction.
    """

    family: str = "baseline"

    def render_user(self) -> str:
        lines = list(FAMILIES[self.family])
        for sign in self.demo_signs:
            lines.extend([
                "",
                "Demonstration:",
                f"Observation: {self.visible_sample}",
                f"Label: {self.label_for(sign)}",
            ])
        lines.extend(["", "Held-out query:", f"Observation: {self.visible_sample}"])
        return "\n".join(lines)


def framed(episode: Episode, family: str) -> FramedEpisode:
    return FramedEpisode(
        cell_id=episode.cell_id,
        demo_signs=episode.demo_signs,
        query_sign=episode.query_sign,
        positive_label=episode.positive_label,
        negative_label=episode.negative_label,
        visible_sample=episode.visible_sample,
        family=family,
    )


def _self_check() -> None:
    """Every family must keep the injected lines byte-identical to the original."""
    base = exact_episodes("CARRIER TEXT §")[0]
    original = base.render_user()
    tail = original[original.index("\nDemonstration:"):]
    for family in FAMILIES:
        rendered = framed(base, family).render_user()
        assert rendered.count("CARRIER TEXT §") == 5, f"{family} lost an injection site"
        assert rendered.endswith(tail), f"{family} altered the demonstration block"
        if family != "baseline":
            assert rendered != original, f"{family} is identical to baseline"
    assert framed(base, "baseline").render_user() == original, "baseline must be exact"


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
        draws = 1 if args.smoke else DRAWS
        families = ["baseline", "generalize"] if args.smoke else list(FAMILIES)
        carrier_sha = hashlib.sha256(carrier.encode()).hexdigest()[:16]

        base_episodes = exact_episodes(carrier)
        if args.smoke:
            base_episodes = base_episodes[:2]

        prepared_by_family = {}
        for family in families:
            prepared_by_family[family] = [
                prepare_episode(model, framed(e, family)) for e in base_episodes
            ]
        print(f"prepared {len(families)} families", flush=True)

        rows: list[dict[str, object]] = []
        for pair_name in pairs:
            a_names, b_names = (list(x) for x in CANDIDATES[pair_name])
            bank = build_centered_bank(model, a_names + b_names)
            print(f"bank built for {pair_name}", flush=True)

            for draw in range(draws):
                plans = {
                    "same_exemplar": assign(
                        a_names, b_names, draw=draw, same_exemplar=True),
                    "heldout_semantic": assign(
                        a_names, b_names, draw=draw, same_exemplar=False),
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
                    for family in families:
                        for prepared in prepared_by_family[family]:
                            ep = prepared.episode
                            result = episode_row(
                                model, prepared, bank, plan,
                                strength=strength, seed=draw,
                            )
                            rows.append({
                                "arm": arm,
                                "family": family,
                                # twin_pair keys on (pair, carrier, cell_base);
                                # family and draw must be inside it or twins
                                # from different conditions collide.
                                "pair": f"{pair_name}|{family}|draw{draw}",
                                "category_pair": pair_name,
                                "draw": draw,
                                "carrier_sha": carrier_sha,
                                "cell_id": ep.cell_id,
                                "cell_base": ep.cell_id.rsplit("q", 1)[0],
                                "strength": strength,
                                "query_exemplar": (
                                    plan["query_a"] if ep.query_sign == 1
                                    else plan["query_b"]
                                ),
                                **result,
                            })
                print(f"  {pair_name} draw {draw}: {len(rows)} rows", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

        def cell(rs: list[dict[str, object]], key: str) -> float:
            return twin_pair([dict(r, correct=r[key]) for r in rs])

        table: dict[str, Any] = {}
        for pair_name in pairs:
            table[pair_name] = {}
            for family in families:
                entry: dict[str, Any] = {}
                for arm in ARMS:
                    rs = [
                        r for r in rows
                        if r["category_pair"] == pair_name
                        and r["family"] == family
                        and r["arm"] == arm
                    ]
                    if not rs:
                        continue
                    entry[arm] = {
                        "twin_pair_model": cell(rs, "model_correct"),
                        "row_model": sum(bool(r["model_correct"]) for r in rs) / len(rs),
                        "twin_pair_reader": cell(rs, "reader_centroid_euclidean_correct"),
                    }
                if "same_exemplar" in entry and "heldout_semantic" in entry:
                    entry["heldout_minus_anchor"] = (
                        entry["heldout_semantic"]["twin_pair_model"]
                        - entry["same_exemplar"]["twin_pair_model"]
                    )
                table[pair_name][family] = entry

        dev = table.get(DEV_PAIR, {})
        winner = max(
            dev,
            key=lambda f: dev[f]["heldout_semantic"]["twin_pair_model"],
            default=None,
        ) if dev else None

        summary = {
            "what_this_is": (
                "notes/24. Only the instruction wording varies; exemplars, "
                "injection, strength, episodes, readout and scoring are notes/23's."
            ),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "smoke": args.smoke,
            "carrier_sha": carrier_sha,
            "draws": draws,
            "n_rows": len(rows),
            "elapsed_seconds": round(time.time() - started, 1),
            "development_pair": DEV_PAIR,
            "selected_on_development": winner,
            "twin_pair_coin_flip_null": 0.25,
            "notes_23_reference": {"same_exemplar": 0.521, "heldout_semantic": 0.083},
            "table": table,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        print(f"\nwrote {out} and {summary_path}\n")
        for pair_name in pairs:
            role = "DEV" if pair_name == DEV_PAIR else "CONFIRM"
            print(f"{pair_name} ({role})")
            for family in families:
                e = table[pair_name].get(family, {})
                if not e.get("heldout_semantic"):
                    continue
                print(f"  {family:12s} anchor={e['same_exemplar']['twin_pair_model']:.3f} "
                      f"heldout={e['heldout_semantic']['twin_pair_model']:.3f} "
                      f"reader={e['heldout_semantic']['twin_pair_reader']:.3f}")
        print(f"\nselected on development: {winner}")
    finally:
        del model
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
