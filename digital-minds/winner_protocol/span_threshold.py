#!/usr/bin/env python3
"""How much steered span does the causal payload need to survive to a readout?

R18 left one number undetermined and the whole family hangs on it. One steered
position moves the naive readout by 0.00; 104 steered positions move it by 3.96.
Both facts are established (`debug_source_effect_L22.jsonl`). The question is
which property of the 104 mattered — *mass*, or *proximity to the readout*.

The crux is `pre_marker`: mass, but distal, with the question span untouched.

* `pre_marker` moves the readout -> the state propagates, so a demonstration span
  can carry it, and R4 is repairable.
* only the readout-adjacent conditions move it -> the state is a local generation
  bias with no propagated trace. Then no prompt can give four demonstrations and
  one query *distinct* recoverable states, and the in-context injected-codebook
  family is closed at the family level rather than at this instance.

DEV, one carrier, neutral, L22, factor 2, frozen source convention. Descriptive:
this localizes an instrument, it does not estimate an effect.

    python span_threshold.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from naive_semantic import render_naive_prompt, marker_position
from protocol_core import MARKER
from welfare_intervention import PublishedActAdd, load_hf_vector
from run_dev import load_model, DEV_CARRIERS, HF_VECTOR_REPO
from debug_source_effect import rating

def marker_turn_span(tok, prompt: str, marker: int) -> tuple[int, ...]:
    """Token positions covering the assistant message that holds the marker."""
    start_char = prompt.index(MARKER)
    encoded = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    first = next(i for i, (a, b) in enumerate(offsets) if b > start_char)
    return tuple(range(first, marker + 1))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vector-file", default="concept_vectors/qwen3-4b_step400/goal/mean_diff.pt")
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--factor", type=float, default=2.0)
    p.add_argument("--out", type=Path, default=Path("results/span_threshold_L22.jsonl"))
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    model, tok, blocks = load_model(a.model)
    vector = load_hf_vector(repo_id=HF_VECTOR_REPO, filename=a.vector_file,
                            layer=a.layer, position=0)
    prompt = render_naive_prompt(tok, DEV_CARRIERS[0], "neutral")
    n = int(tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.shape[1])
    marker = marker_position(tok, prompt)
    turn = marker_turn_span(tok, prompt, marker)

    conditions = {
        "marker_only": (marker,),
        "marker_turn": turn,
        "pre_marker": tuple(range(0, marker + 1)),
        "post_marker": tuple(range(marker, n)),
        "last_4": tuple(range(n - 4, n)),
        "last_16": tuple(range(n - 16, n)),
        "all": tuple(range(n)),
    }

    baseline = rating(model, tok, blocks, prompt, None)
    print(f"{n} tokens, marker {marker}, marker turn {turn[0]}..{turn[-1]}, "
          f"baseline {baseline:.3f}")
    print(f"{'condition':>12} {'k':>4}  {'-v':>6} {'+v':>6}  {'delta':>7}")

    rows = []
    for name, positions in conditions.items():
        by_sign = {}
        for sign in (-1, +1):
            edit = PublishedActAdd(
                layer=a.layer, vector=vector, factor=a.factor,
                positions=positions, signs=(sign,) * len(positions),
            )
            by_sign[sign] = rating(model, tok, blocks, prompt, edit)
        delta = by_sign[+1] - by_sign[-1]
        rows.append({
            "condition": name, "n_edited": len(positions),
            "first_pos": positions[0], "last_pos": positions[-1],
            "includes_readout": positions[-1] == n - 1,
            "layer": a.layer, "factor": a.factor, "baseline": baseline,
            "rating_minus": by_sign[-1], "rating_plus": by_sign[+1],
            "plus_minus": delta,
        })
        print(f"{name:>12} {len(positions):>4}  {by_sign[-1]:>6.3f} "
              f"{by_sign[+1]:>6.3f}  {delta:>+7.3f}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    distal = next(r for r in rows if r["condition"] == "pre_marker")
    print(f"\ncrux — pre_marker (mass {distal['n_edited']}, readout untouched): "
          f"{distal['plus_minus']:+.3f}")
    print("propagates; R4 repairable" if abs(distal["plus_minus"]) >= 0.5
          else "no propagated trace; family-level closure for injected codebooks")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
