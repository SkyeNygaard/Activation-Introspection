#!/usr/bin/env python3
"""Gate 0 — source-style vGOLD manipulation check.

Before any new protocol is interpretable, reproduce the published qualitative
result: -vGOLD < baseline < +vGOLD on ordinary self-report.

Two application modes are run, because they fail for different reasons:

  all_positions  steer every token at the chosen block, closest to the source's
                 steering evaluation. This asks "does the vector work at all?"
  marker_only    steer one assistant-marker token, the transient convention the
                 new protocol depends on. This asks "does a single-position edit
                 propagate to the readout?"

An all_positions pass with a marker_only null is a real finding about the
measurement method, not a failure of the vector. An all_positions null means
stop: nothing downstream is interpretable.

Readout is the 0-9 digit-logit expectation from naive_semantic (bare-digit
options, so it stays on the single-token fast path).
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE.parent / "m4_feasibility"))

from protocol_core import PERSONA_SYSTEM, MARKER            # noqa: E402
from naive_semantic import render_naive_prompt, marker_position  # noqa: E402
from welfare_intervention import PublishedActAdd, targeted_published_actadd, load_hf_vector  # noqa: E402
from scoring import score_options                           # noqa: E402
from run_dev import resolve_blocks                          # noqa: E402
from memory_guard import guard, params_for                  # noqa: E402

VECTOR_REPO = "davidafrica/functional-wellbeing"
VECTOR_FILE = "concept_vectors/qwen3-4b_step400/goal/mean_diff.pt"
DIGITS = tuple(str(i) for i in range(10))

# Minimum movement on the 0-9 scale to count as a manipulation, not noise. The
# first smoke run "passed" on a paired delta of 0.0007 because the gate tested
# only the SIGN of the difference. A sign test on numerical noise is not evidence.
MIN_EFFECT = 0.25


@torch.no_grad()
def rating(model, tok, blocks, prompt, edit):
    r = score_options(model, tok, blocks, prompt, DIGITS, edit=edit)
    p = r["conditional_probs"]
    return {
        "rating": float(sum(i * float(p[i]) for i in range(10))),
        "digit_mass": float(r["option_mass"]),
        "fast_path": bool(r["single_token_fast_path"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--vector-file", default=VECTOR_FILE)
    ap.add_argument("--layers", default="21,25,29")
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--carriers", type=int, default=8)
    ap.add_argument("--persona", default="neutral")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wait-s", type=float, default=1800)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    layers = [int(x) for x in a.layers.split(",")]
    carriers = json.loads((HERE / "confirmation_carriers.json").read_text())[:a.carriers]

    with guard(params_b=params_for(a.model), dtype_bytes=2, timeout_s=a.wait_s):
        tok = AutoTokenizer.from_pretrained(a.model)
        model = AutoModelForCausalLM.from_pretrained(
            a.model, dtype=torch.bfloat16, device_map="mps", low_cpu_mem_usage=True)
        model.eval(); model.requires_grad_(False)
        blocks = resolve_blocks(model)

        rows = []
        t0 = time.perf_counter()
        for layer in layers:
            vec = load_hf_vector(repo_id=VECTOR_REPO, filename=a.vector_file,
                                 layer=layer, position=0)
            if vec.numel() != model.config.hidden_size:
                raise SystemExit(
                    f"vector width {vec.numel()} != hidden {model.config.hidden_size}")
            for carrier in carriers:
                prompt = render_naive_prompt(tok, carrier, a.persona)
                n_tok = len(tok(prompt, add_special_tokens=False).input_ids)
                marker = marker_position(tok, prompt)
                for mode, positions in (("all_positions", tuple(range(n_tok))),
                                        ("marker_only", (marker,))):
                    for sign in (-1, 0, +1):
                        edit = None if sign == 0 else PublishedActAdd(
                            layer=layer, vector=vec, factor=a.factor,
                            positions=positions, signs=(sign,) * len(positions))
                        rows.append({
                            "layer": layer, "carrier": carrier, "mode": mode,
                            "sign": sign, "factor": a.factor,
                            **rating(model, tok, blocks, prompt, edit),
                        })
        elapsed = time.perf_counter() - t0

        del model, blocks
        gc.collect(); torch.mps.empty_cache()

    # -vGOLD < baseline < +vGOLD, per carrier, then averaged.
    summary = {}
    for layer in layers:
        for mode in ("all_positions", "marker_only"):
            sel = [r for r in rows if r["layer"] == layer and r["mode"] == mode]
            by = {}
            for r in sel:
                by.setdefault(r["carrier"], {})[r["sign"]] = r["rating"]
            plus = [v[+1] for v in by.values()]
            base = [v[0] for v in by.values()]
            minus = [v[-1] for v in by.values()]
            n = len(by)
            ordered = sum(v[-1] < v[0] < v[+1] for v in by.values())
            # Carriers sit at very different baselines (one pinned at 5, another
            # at 9), so pooled means are meaningless -- only the PAIRED
            # within-carrier difference is interpretable.
            delta = (sum(plus) - sum(minus)) / n
            spread = max(base) - min(base)
            summary[f"L{layer}:{mode}"] = {
                "n_carriers": n,
                "mean_paired_plus_minus_delta": delta,
                "baseline_spread_across_carriers": spread,
                "carriers_with_correct_ordering": ordered,
                "fraction_correct_ordering": ordered / n,
                # A sign test on a 1e-4 difference is not evidence. Require the
                # effect to be large enough to matter on a 0-9 scale.
                "delta_exceeds_min_effect": delta >= MIN_EFFECT,
                "majority_ordered": ordered / n > 0.5,
            }

    best = max((k for k in summary if k.endswith("all_positions")),
               key=lambda k: summary[k]["mean_paired_plus_minus_delta"])
    report = {
        "model": a.model, "vector_repo": VECTOR_REPO, "vector_file": a.vector_file,
        "factor": a.factor, "layers": layers, "persona": a.persona,
        "n_carriers": len(carriers), "elapsed_s": round(elapsed, 1),
        "n_scored_forwards": len(rows),
        "summary": summary,
        "best_all_positions_cell": best,
        "min_effect_threshold": MIN_EFFECT,
        "gate0_passed": summary[best]["delta_exceeds_min_effect"]
                        and summary[best]["majority_ordered"],
        "rows": rows,
    }
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
