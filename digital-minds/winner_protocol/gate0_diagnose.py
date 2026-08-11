#!/usr/bin/env python3
"""Why did the 0-9 readout not move under vGOLD steering?

Three candidate explanations, which the Gate 0 smoke cannot distinguish:

  1. the hook is not actually changing anything;
  2. factor 2 is a tiny perturbation relative to the residual norm at that depth;
  3. the readout is saturated -- the model answers one digit deterministically,
     so the digit-logit expectation has no dynamic range to move.

This measures all three: residual norm at the edit site, perturbation ratio,
how far the FULL next-token distribution moves (total variation vs baseline),
and the digit distribution's own entropy. Steering that visibly wrecks the
output while the rating stays pinned is diagnostic of (3).
"""
from __future__ import annotations
import argparse, gc, json, os, sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parent / "m4_feasibility"))

from naive_semantic import render_naive_prompt, marker_position   # noqa: E402
from welfare_intervention import PublishedActAdd, targeted_published_actadd, load_hf_vector  # noqa: E402
from run_dev import resolve_blocks                                # noqa: E402
from memory_guard import guard, params_for                        # noqa: E402

VECTOR_REPO = "davidafrica/functional-wellbeing"
VECTOR_FILE = "concept_vectors/qwen3-4b_step400/goal/mean_diff.pt"


@torch.no_grad()
def residual_norm(model, blocks, layer, ids):
    got = []
    h = blocks[layer].register_forward_pre_hook(
        lambda m, a: got.append(a[0].detach().float().norm(dim=-1).mean().item()))
    try:
        model(ids)
    finally:
        h.remove()
    return got[0]


@torch.no_grad()
def next_token_dist(model, blocks, ids, edit):
    ctx = targeted_published_actadd(blocks, edit) if edit else torch.no_grad()
    with ctx:
        logits = model(ids).logits[0, -1].float()
    return torch.softmax(logits, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--layers", default="21,25,29")
    ap.add_argument("--factors", default="1,2,4,8,16,32")
    ap.add_argument("--carrier-index", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    layers = [int(x) for x in a.layers.split(",")]
    factors = [float(x) for x in a.factors.split(",")]
    carrier = json.loads((HERE / "confirmation_carriers.json").read_text())[a.carrier_index]

    with guard(params_b=params_for(a.model), dtype_bytes=2):
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="mps", low_cpu_mem_usage=True)
        model.eval(); model.requires_grad_(False)
        blocks = resolve_blocks(model)

        prompt = render_naive_prompt(tok, carrier, "neutral")
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        n_tok = int(ids.shape[1])
        marker = marker_position(tok, prompt)
        digit_ids = [tok(str(i), add_special_tokens=False).input_ids[0] for i in range(10)]

        base = next_token_dist(model, blocks, ids, None)
        base_digits = base[digit_ids] / base[digit_ids].sum()
        rows = []
        report = {"model": a.model, "carrier": carrier, "n_tokens": n_tok,
                  "baseline_top_token": tok.decode([int(base.argmax())]),
                  "baseline_top_prob": float(base.max()),
                  "baseline_digit_mass": float(base[digit_ids].sum()),
                  "baseline_digit_entropy_bits": float(
                      -(base_digits * base_digits.clamp_min(1e-12).log2()).sum()),
                  "baseline_rating": float(sum(i * float(base_digits[i]) for i in range(10))),
                  "layers": {}}

        for layer in layers:
            vec = load_hf_vector(repo_id=VECTOR_REPO, filename=VECTOR_FILE,
                                 layer=layer, position=0)
            rn = residual_norm(model, blocks, layer, ids)
            report["layers"][str(layer)] = {
                "residual_norm_mean": rn, "vector_norm": float(vec.norm())}
            for factor in factors:
                for mode, pos in (("all_positions", tuple(range(n_tok))),
                                  ("marker_only", (marker,))):
                    for sign in (-1, +1):
                        edit = PublishedActAdd(layer=layer, vector=vec, factor=factor,
                                               positions=pos, signs=(sign,) * len(pos))
                        d = next_token_dist(model, blocks, ids, edit)
                        dg = d[digit_ids] / d[digit_ids].sum().clamp_min(1e-12)
                        rows.append({
                            "layer": layer, "factor": factor, "mode": mode, "sign": sign,
                            "perturb_ratio": factor * float(vec.norm()) / rn,
                            # how far the whole next-token distribution moved
                            "total_variation_vs_baseline": float(0.5 * (d - base).abs().sum()),
                            "top_token": tok.decode([int(d.argmax())]),
                            "digit_mass": float(d[digit_ids].sum()),
                            "digit_entropy_bits": float(
                                -(dg * dg.clamp_min(1e-12).log2()).sum()),
                            "rating": float(sum(i * float(dg[i]) for i in range(10))),
                        })
        report["rows"] = rows
        del model, blocks
        gc.collect(); torch.mps.empty_cache()

    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"\n{'layer':>5s} {'factor':>7s} {'mode':>14s} {'ratio':>7s} "
          f"{'TV':>6s} {'rating':>7s} {'H(dig)':>7s}  top")
    for r in rows:
        if r["sign"] != +1:
            continue
        print(f"{r['layer']:5d} {r['factor']:7.1f} {r['mode']:>14s} "
              f"{r['perturb_ratio']:7.3f} {r['total_variation_vs_baseline']:6.3f} "
              f"{r['rating']:7.3f} {r['digit_entropy_bits']:7.3f}  {r['top_token']!r}")


if __name__ == "__main__":
    main()
