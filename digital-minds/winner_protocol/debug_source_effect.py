#!/usr/bin/env python3
"""The one bounded debugging pass allowed by GPU_RUN_ORDER step 0.

The naive readout showed no `+vGOLD` / `-vGOLD` effect at L29, factor 2, edited
at the single assistant marker token. Two explanations survive that result, and
they lead to opposite decisions:

* the published vector is inert in our hands -> kill the public-vector branch;
* a one-token edit is too narrow to carry, while the source convention steers
  every position -> the vector is fine and the *protocol's* marker edit is what
  fails to be load-bearing.

So this varies scope and factor together, and reports the edit magnitude against
the residual norm it is competing with. It deliberately does not vary anything
else: no layers, no prompts, no personas.

    python debug_source_effect.py --layer 29
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from naive_semantic import render_naive_prompt, marker_position
from scoring import score_options
from welfare_intervention import PublishedActAdd, load_hf_vector
from run_dev import load_model, DEV_CARRIERS, HF_VECTOR_REPO

def residual_norms(model, blocks, ids, layer: int) -> torch.Tensor:
    """Norm of each position's residual at the input to `blocks[layer]`."""
    captured = {}

    def hook(_module, args):
        captured["hidden"] = args[0].detach()

    handle = blocks[layer].register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return captured["hidden"][0].float().norm(dim=-1)

def rating(model, tok, blocks, prompt, edit) -> float:
    result = score_options(
        model, tok, blocks, prompt, tuple(f" {i}" for i in range(10)), edit=edit
    )
    probs = result["conditional_probs"]
    return float(sum(i * float(p) for i, p in enumerate(probs)))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vector-file", default="concept_vectors/qwen3-4b_step400/goal/mean_diff.pt")
    p.add_argument("--layer", type=int, default=29)
    p.add_argument("--factors", default="2,8,16,32")
    p.add_argument("--out", type=Path, default=Path("results/debug_source_effect_L29.jsonl"))
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    model, tok, blocks = load_model(a.model)
    vector = load_hf_vector(repo_id=HF_VECTOR_REPO, filename=a.vector_file,
                            layer=a.layer, position=0)

    carrier = DEV_CARRIERS[0]
    prompt = render_naive_prompt(tok, carrier, "neutral")
    marker = marker_position(tok, prompt)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    norms = residual_norms(model, blocks, ids, a.layer)
    n_tokens = int(ids.shape[1])

    print(f"tokens {n_tokens}, marker at {marker}")
    print(f"||vector|| {float(vector.norm()):.2f}")
    print(f"residual at L{a.layer}: marker {float(norms[marker]):.1f}, "
          f"median {float(norms.median()):.1f}, last {float(norms[-1]):.1f}")

    baseline = rating(model, tok, blocks, prompt, None)
    print(f"baseline rating {baseline:.3f}")

    rows = []
    for factor in [float(x) for x in a.factors.split(",")]:
        for scope in ("marker", "all"):
            positions = (marker,) if scope == "marker" else tuple(range(n_tokens))
            by_sign = {}
            for sign in (-1, +1):
                edit = PublishedActAdd(
                    layer=a.layer, vector=vector, factor=factor,
                    positions=positions, signs=(sign,) * len(positions),
                )
                by_sign[sign] = rating(model, tok, blocks, prompt, edit)
            row = {
                "layer": a.layer, "factor": factor, "scope": scope,
                "n_edited": len(positions),
                "rating_minus": by_sign[-1], "rating_plus": by_sign[+1],
                "plus_minus": by_sign[+1] - by_sign[-1],
                "baseline": baseline,
                "edit_norm_over_marker_residual": factor * float(vector.norm()) / float(norms[marker]),
            }
            rows.append(row)
            print(f"factor {factor:>5g} {scope:>6}: -v {row['rating_minus']:.3f}  "
                  f"+v {row['rating_plus']:.3f}  delta {row['plus_minus']:+.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
