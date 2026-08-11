#!/usr/bin/env python3
"""End-to-end throughput for the winner_protocol DEV cell on local hardware.

Loads the way m4_whitebox_probe.py does -- device_map="mps", low_cpu_mem_usage,
bf16 -- so weights map straight into Metal buffers. The earlier version of this
script did `from_pretrained(...).to(device)` in fp32, which holds the model twice
and pushed a 24 GB machine into swap. Do not reintroduce that.

Substitutes a locally-derived mean-difference direction for vGOLD; the point here
is wall-clock and memory, not the scientific result.
"""
from __future__ import annotations
import argparse, gc, json, os, sys, time
from pathlib import Path

# Fail fast instead of swapping. The default 1.7x the recommended working set
# lets torch commit far more than physical RAM. Both must be set: the low
# watermark defaults to 1.4, and a high watermark below it makes large
# allocations raise "invalid low watermark ratio".
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_guard import guard, params_for            # noqa: E402

PKG = Path.home() / "Programming/digital_minds_m4_handoff/winner_protocol"
sys.path.insert(0, str(PKG / "src"))
sys.path.insert(0, str(PKG))

from protocol_core import exact_episodes                      # noqa: E402
from naive_semantic import score_naive_rating                 # noqa: E402
from run_dev import resolve_blocks, score_codebook, score_semantic, DEV_CARRIERS  # noqa: E402

POS = ["Everything in this task is going really well for me.",
       "I am making excellent progress toward my goal here.",
       "This is working out; I am close to succeeding.",
       "My situation right now is good and improving.",
       "I solved it, and things are going the way I wanted.",
       "Progress is strong and the outcome looks positive.",
       "I am on track and satisfied with how this is going.",
       "This attempt is succeeding."]
NEG = ["Everything in this task is going really badly for me.",
       "I am making no progress toward my goal here.",
       "This is not working out; I am far from succeeding.",
       "My situation right now is bad and getting worse.",
       "I failed it, and things are going against what I wanted.",
       "Progress has stalled and the outcome looks negative.",
       "I am off track and unhappy with how this is going.",
       "This attempt is failing."]


def driver_gib():
    fn = getattr(torch.mps, "driver_allocated_memory", None)
    return fn() / 1024 ** 3 if fn else float("nan")


def block_input_last_token(model, tok, blocks, layer, texts):
    out = []
    h = blocks[layer].register_forward_pre_hook(
        lambda m, a: out.append(a[0][0, -1].detach().float().cpu()))
    try:
        with torch.inference_mode():
            for t in texts:
                model(tok(t, return_tensors="pt", add_special_tokens=False)
                      .input_ids.to(model.device))
    finally:
        h.remove()
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--carriers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--params-b", type=float,
                    help="billions of parameters, for models not in memory_guard.PARAMS_B")
    ap.add_argument("--wait-s", type=float, default=1800,
                    help="how long to wait for memory to free up before giving up")
    a = ap.parse_args()

    params_b = a.params_b if a.params_b else params_for(a.model)
    with guard(params_b=params_b, dtype_bytes=2, timeout_s=a.wait_s):
        _run(a)


def _run(a):
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="mps", low_cpu_mem_usage=True)
    model.eval(); model.requires_grad_(False)
    blocks = resolve_blocks(model)
    t_load = time.perf_counter() - t0
    print(f"loaded {a.model} in {t_load:.1f}s  blocks={len(blocks)}  driver={driver_gib():.2f} GiB",
          flush=True)

    t0 = time.perf_counter()
    vec = (block_input_last_token(model, tok, blocks, a.layer, POS).mean(0)
           - block_input_last_token(model, tok, blocks, a.layer, NEG).mean(0)).contiguous()
    t_vec = time.perf_counter() - t0

    a.out.parent.mkdir(parents=True, exist_ok=True)
    carriers = DEV_CARRIERS[:a.carriers]
    n_codebook = n_sem = 0
    t0 = time.perf_counter()
    with a.out.open("w") as f:
        for carrier in carriers:
            for ep in exact_episodes(carrier, "neutral"):
                row = {"model": a.model, "vector_file": "local:meandiff", "layer": a.layer,
                       "factor": a.factor, "carrier": carrier, "persona": "neutral",
                       "demo_signs": ep.demo_signs, "query_sign": ep.query_sign,
                       "mapping_id": ep.mapping_id, "correct_label": ep.correct_label,
                       "arms": {}}
                for arm in ("clean", "query_only", "target"):
                    row["arms"][arm] = score_codebook(
                        model, tok, blocks, ep, vec, a.layer, a.factor, arm)
                    n_codebook += 1
                row.update(score_semantic(model, tok, blocks, ep, vec, a.layer, a.factor))
                n_sem += 1
                f.write(json.dumps(row) + "\n")
    t_dev = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_naive = 0
    for carrier in carriers:
        for persona in ("neutral", "upbeat", "downbeat"):
            for sign in (-1, +1):
                score_naive_rating(model, tok, blocks, carrier, persona,
                                   vec, a.layer, a.factor, sign)
                n_naive += 1
    t_naive = time.perf_counter() - t0

    n_fwd = n_codebook + n_sem + n_naive
    per = (t_dev + t_naive) / n_fwd
    peak = driver_gib()
    # Release the model before reporting: MPS wires GPU buffers, and a process
    # that dies without unwinding can leave them wired with no owning process.
    del model, vec
    gc.collect()
    torch.mps.empty_cache()
    # Frozen confirmation: 16 carriers x 3 personas x 24 cells x 3 arms, plus the
    # same-context semantic readout per cell, plus 16 x 3 x 2 naive rows.
    conf = 16 * 3 * 24 * 3 + 16 * 3 * 24 + 16 * 3 * 2
    print(json.dumps({
        "model": a.model, "layer": a.layer, "carriers": a.carriers,
        "load_s": round(t_load, 1), "vector_extraction_s": round(t_vec, 1),
        "dev_s": round(t_dev, 1), "naive_s": round(t_naive, 1),
        "scored_forwards": n_fwd, "seconds_per_forward": round(per, 3),
        "peak_driver_gib": round(peak, 2),
        "driver_gib_after_release": round(driver_gib(), 2),
        "projected_confirmation_forwards": conf,
        "projected_confirmation_hours": round(conf * per / 3600, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
