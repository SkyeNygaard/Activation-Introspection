#!/usr/bin/env python3
"""Exercise the real model path cheaply, on a small local model.

`preflight.py` is deliberately model-free, so until now nothing in this package
had ever run against weights: the chat template, the marker-position mapping,
the pre-hook and the option scorer were only ever exercised against the
simulator. This runs all of them once on a model small enough to load anywhere,
with a synthetic vector of the right width, so that a scarce 4B slot is spent on
the experiment rather than on discovering a typo.

It checks plumbing, not welfare. A synthetic vector says nothing about vGOLD.

    python smoke_local.py --model Qwen/Qwen2.5-0.5B-Instruct --layer 12
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from naive_semantic import score_naive_rating
from protocol_core import exact_episodes
from run_dev import load_model, score_codebook, score_semantic, DEV_CARRIERS

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--factor", type=float, default=2.0)
    a = p.parse_args()

    model, tok, blocks = load_model(a.model)
    print(f"loaded {a.model} on {model.device}, {len(blocks)} blocks")
    assert a.layer < len(blocks), f"layer {a.layer} outside {len(blocks)}-block model"

    d = model.config.hidden_size
    torch.manual_seed(0)
    vector = torch.randn(d) / d**0.5  # stand-in only; not a welfare direction

    carrier = DEV_CARRIERS[0]
    naive = score_naive_rating(
        model, tok, blocks, carrier, "neutral", vector, a.layer, a.factor, +1
    )
    assert 0.0 <= naive["semantic_rating"] <= 9.0, naive
    print(f"naive rating {naive['semantic_rating']:.2f} "
          f"(fast path {naive['single_token_fast_path']})")

    episode = exact_episodes(carrier, "neutral")[0]
    for arm in ("clean", "query_only", "target"):
        row = score_codebook(model, tok, blocks, episode, vector, a.layer, a.factor, arm)
        assert row["predicted_label"] in ("Q", "K"), row
        assert 0.0 <= row["p_correct"] <= 1.0, row
        print(f"{arm:>10}: pred {row['predicted_label']} "
              f"p_correct {row['p_correct']:.3f} label_mass {row['label_mass']:.3f} "
              f"format_ok {row['format_ok']}")

    sem = score_semantic(model, tok, blocks, episode, vector, a.layer, a.factor)
    assert 0.0 <= sem["semantic_rating"] <= 9.0, sem
    print(f"in-context semantic rating {sem['semantic_rating']:.2f}")

    # A hook left installed would silently contaminate every later arm.
    assert not blocks[a.layer]._forward_pre_hooks, "pre-hook survived its context manager"
    print("hooks clean; local model path OK")

if __name__ == "__main__":
    main()
