#!/usr/bin/env python3
"""Where, if anywhere, does the public vGOLD tensor steer the naive readout?

R17 killed L29 on the source-sanity gate. L29 came from R10, which read it off
the artifact's documentation rather than measuring it, and the cached tensor
carries all 36 layers. This localizes the effect along the vector's own layer
axis under the source convention (every position edited), so that "inert in our
hands" and "R10 named the wrong layer" stop being the same result.

One carrier, neutral persona, both signs, two factors. Nothing else varies: this
measures the artifact, it does not tune the protocol.

    python localize_source_layer.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from naive_semantic import render_naive_prompt
from welfare_intervention import PublishedActAdd, load_hf_vector
from run_dev import load_model, DEV_CARRIERS, HF_VECTOR_REPO
from debug_source_effect import rating

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vector-file", default="concept_vectors/qwen3-4b_step400/goal/mean_diff.pt")
    p.add_argument("--factors", default="2,8")
    p.add_argument("--out", type=Path, default=Path("results/localize_source_layer.jsonl"))
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    model, tok, blocks = load_model(a.model)
    prompt = render_naive_prompt(tok, DEV_CARRIERS[0], "neutral")
    n_tokens = int(tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.shape[1])
    positions = tuple(range(n_tokens))
    factors = [float(x) for x in a.factors.split(",")]

    baseline = rating(model, tok, blocks, prompt, None)
    print(f"{len(blocks)} blocks, {n_tokens} tokens, baseline {baseline:.3f}")
    print("layer  |v|   " + "  ".join(f"f{f:g}: -v/+v/delta" for f in factors))

    rows = []
    for layer in range(len(blocks)):
        vector = load_hf_vector(repo_id=HF_VECTOR_REPO, filename=a.vector_file,
                                layer=layer, position=0)
        line = f"{layer:>5} {float(vector.norm()):>5.1f}  "
        for factor in factors:
            by_sign = {}
            for sign in (-1, +1):
                edit = PublishedActAdd(
                    layer=layer, vector=vector, factor=factor,
                    positions=positions, signs=(sign,) * n_tokens,
                )
                by_sign[sign] = rating(model, tok, blocks, prompt, edit)
            rows.append({
                "layer": layer, "factor": factor, "scope": "all",
                "vector_norm": float(vector.norm()),
                "rating_minus": by_sign[-1], "rating_plus": by_sign[+1],
                "plus_minus": by_sign[+1] - by_sign[-1], "baseline": baseline,
            })
            line += (f" {by_sign[-1]:>5.2f}/{by_sign[+1]:>5.2f}/"
                     f"{by_sign[+1] - by_sign[-1]:>+6.3f} ")
        print(line, flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    best = max(rows, key=lambda r: r["plus_minus"])
    print(f"largest delta {best['plus_minus']:+.3f} at L{best['layer']} "
          f"factor {best['factor']:g} (persona effect for scale: 6.13)")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
