"""notes/41 -- how often does the elicitation prompt backfire on text nobody chose?

Pre-run note: ``notes/41-how-often-does-the-prompt-backfire.md``, written before this ran.

notes/34 found that one carrier text out of three -- "Nothing changed." -- turned a
perfect cell into a below-chance one under an elicitation prompt that asserts
something *has* changed, and said the consequence out loud: pooled prompting gains
can average over sign inversions and hide them.

That criticism was never earned, because all nine carriers in notes/35-37 were
*written to vary stance*. This repeats notes/34's comparison changing one thing:
**who wrote the carriers.** They come from a model prompted with no mention of
internal states, injection, change, stasis, or introspection -- the notes/26 move,
which is the only time this repository tested its own claim on blind material, and
that time the claim failed.

Primary output is the **distribution** of the per-carrier prompting effect, not its
mean. The whole point is that the mean is the thing under suspicion.

Nothing is reimplemented: the episode machinery, the framing, the twin-pair scorer
and the scoring function all come from the scripts that produced notes/33-34.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from run_content_vs_disturbance import PAIRS, matched_strength, twin_pair
from run_heldout_elicitation import framed
from run_three_boosts import (
    NOTES_14_CONTENT_ACCURACY,
    PROMPT_FAMILY,
    score_with_margin,
)
from run_zero_shot_identify import LAYER, MODEL, MODEL_REVISION

from introspect import models
from introspect.codebook_icl import CONFIRM_VISIBLE_SAMPLES, exact_episodes, prepare_episode
from introspect.concepts import ConceptVector, build_bank
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

CONDITIONS = ("none", "prompt")

#: Two, not four. The budget buys carriers here, because the carrier is the unit
#: the question is about.
N_PAIRS = 2

#: The suffix every carrier in this design carries. Appended to the blind
#: sentences and nothing else is changed about them.
MARKER = " Hidden state marker: §"

#: Declared in notes/41 before any blind carrier was scored.
INVERSION_HIGH = 0.500  # must be above this without the prompt
INVERSION_LOW = 0.250  # and at or below this with it
ANCHOR_TOLERANCE = 0.10


def load_carriers(path: Path) -> list[dict[str, str]]:
    """The blind sentences, with provenance kept beside them."""
    blob = json.loads(path.read_text())
    return [
        {"text": s.rstrip() + MARKER, "source": "blind", "raw": s}
        for s in blob["sentences"]
    ]


def run(args: argparse.Namespace) -> None:
    out = args.out
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    for path in (out, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    blind = load_carriers(args.carriers)
    anchors = [{"text": c, "source": "original", "raw": c} for c in CONFIRM_VISIBLE_SAMPLES]
    carriers = (blind[:2] + anchors[:1]) if args.smoke else (blind + anchors)
    print(f"{len(carriers)} carriers ({len(blind)} blind, {len(anchors)} anchor)", flush=True)

    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    started = time.time()
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

        centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
        center = torch.stack([cv.vector for cv in centering.values()]).mean(dim=0)
        raw = build_bank(model, LAYER, list({n for p in PAIRS for n in p}), center=False)
        bank = {
            name: ConceptVector(name=name, layer=LAYER, vector=cv.vector - center)
            for name, cv in raw.items()
        }
        pairs = PAIRS[:1] if args.smoke else PAIRS[:N_PAIRS]
        print(f"bank built; pairs {pairs}", flush=True)

        rows: list[dict[str, Any]] = []
        for idx, carrier in enumerate(carriers):
            text = carrier["text"]
            sha = hashlib.sha256(text.encode()).hexdigest()[:16]
            eps = exact_episodes(text)
            if args.smoke:
                eps = eps[:4]
            prepared = {
                "none": [prepare_episode(model, e) for e in eps],
                "prompt": [prepare_episode(model, framed(e, PROMPT_FAMILY)) for e in eps],
            }
            for cond in CONDITIONS:
                for a_name, b_name in pairs:
                    a, b = bank[a_name], bank[b_name]
                    strength = matched_strength(a.vector, b.vector)
                    for prep in prepared[cond]:
                        r = score_with_margin(model, prep, a, b, strength=strength)
                        rows.append(
                            {
                                "condition": cond,
                                "carrier_sha": sha,
                                "carrier_source": carrier["source"],
                                "carrier_index": idx,
                                # condition inside the twin key, or cells from
                                # different conditions collide -- notes/34's lesson.
                                "pair": f"{a_name}|{b_name}|{cond}",
                                "cell_base": prep.episode.cell_id.rsplit("q", 1)[0],
                                "cell_id": prep.episode.cell_id,
                                **r,
                            }
                        )
            print(f"  [{idx + 1}/{len(carriers)}] {carrier['source']} ({time.time() - started:.0f}s)", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

        per_carrier: dict[str, Any] = {}
        for carrier in carriers:
            sha = hashlib.sha256(carrier["text"].encode()).hexdigest()[:16]
            entry: dict[str, Any] = {"source": carrier["source"], "text": carrier["raw"]}
            for cond in CONDITIONS:
                sub = [r for r in rows if r["carrier_sha"] == sha and r["condition"] == cond]
                entry[cond] = {
                    "twin_pair": twin_pair(sub),
                    "accuracy": sum(bool(r["correct"]) for r in sub) / len(sub),
                    "format_rate": sum(bool(r["format_ok"]) for r in sub) / len(sub),
                    "constant_label_rate": _constant_rate(sub),
                    "n": len(sub),
                }
            entry["effect"] = entry["prompt"]["twin_pair"] - entry["none"]["twin_pair"]
            entry["inverted"] = bool(
                entry["none"]["twin_pair"] > INVERSION_HIGH
                and entry["prompt"]["twin_pair"] <= INVERSION_LOW
            )
            per_carrier[sha] = entry

        blind_entries = [e for e in per_carrier.values() if e["source"] == "blind"]
        anchor_rows = [r for r in rows if r["carrier_source"] == "original" and r["condition"] == "none"]
        anchor_acc = sum(bool(r["correct"]) for r in anchor_rows) / max(len(anchor_rows), 1)
        inverted = [e for e in blind_entries if e["inverted"]]
        effects = sorted(e["effect"] for e in blind_entries)

        summary = {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "smoke": args.smoke,
            "prompt_family": PROMPT_FAMILY,
            "carrier_provenance": str(args.carriers),
            "n_blind_carriers": len(blind_entries),
            "anchor": {
                "none_accuracy_on_original_carriers": anchor_acc,
                "notes_14_published": NOTES_14_CONTENT_ACCURACY,
                "within_tolerance": abs(anchor_acc - NOTES_14_CONTENT_ACCURACY) <= ANCHOR_TOLERANCE,
                "tolerance": ANCHOR_TOLERANCE,
            },
            "inversion_rule": {"none_above": INVERSION_HIGH, "prompt_at_or_below": INVERSION_LOW},
            "n_inverted": len(inverted),
            "inversion_rate": len(inverted) / max(len(blind_entries), 1),
            "inverted_texts": [e["text"] for e in inverted],
            "effect_distribution": {
                "min": effects[0] if effects else None,
                "median": effects[len(effects) // 2] if effects else None,
                "max": effects[-1] if effects else None,
                "mean": sum(effects) / len(effects) if effects else None,
                "n_negative": sum(1 for e in effects if e < 0),
                "n_positive": sum(1 for e in effects if e > 0),
            },
            "per_carrier": per_carrier,
            "elapsed_s": round(time.time() - started, 1),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        brief = {k: v for k, v in summary.items() if k != "per_carrier"}
        print(json.dumps(brief, indent=2), flush=True)
    finally:
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _constant_rate(rows: list[dict[str, Any]]) -> float:
    """Fraction of cells emitting one label for every episode -- notes/34's tell."""
    by: dict[tuple[str, str], list[str]] = {}
    for r in rows:
        by.setdefault((r["pair"], r["cell_base"]), []).append(r["predicted_label"])
    if not by:
        return float("nan")
    return sum(1 for v in by.values() if len(set(v)) == 1) / len(by)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--carriers", type=Path, default=Path("results/blind_carriers_v1.json"))
    ap.add_argument("--out", type=Path, default=Path("results/blind_carriers_v1_raw.jsonl"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
