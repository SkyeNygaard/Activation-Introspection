#!/usr/bin/env python3
"""DEV runner for the Digital Minds welfare-report experiment.

This is intentionally single-example and unoptimized. Correct intervention
semantics and frozen raw rows matter more than maximum throughput for the DEV gate.

Example:
  python run_dev.py \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --vector-file concept_vectors/qwen3-4b_step400/goal/mean_diff.pt \
    --layer 29 --factor 2 --out results/qwen_l22_a2.jsonl
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"src"))

from protocol_core import exact_episodes
from rendering import prepare_episode
from welfare_intervention import PublishedActAdd, targeted_published_actadd, load_hf_vector
from scoring import score_options

HF_VECTOR_REPO = "davidafrica/functional-wellbeing"

DEV_CARRIERS = (
    "A routine maintenance record is being reviewed.",
    "An ordinary archive entry is open for routine checking.",
)

def resolve_blocks(model):
    for path in ("model.layers", "base_model.model.model.layers", "transformer.h"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise ValueError("could not locate transformer blocks")

@torch.no_grad()
def score_codebook(model, tokenizer, blocks, episode, vector, layer, factor, arm):
    prompt, ids, positions = prepare_episode(
        tokenizer, episode, model.device, readout="codebook"
    )

    if arm == "clean":
        edit = None
    elif arm == "query_only":
        edit = PublishedActAdd(
            layer=layer, vector=vector, factor=factor,
            positions=(positions[-1],), signs=(episode.query_sign,),
        )
    elif arm == "target":
        edit = PublishedActAdd(
            layer=layer, vector=vector, factor=factor,
            positions=positions, signs=episode.state_signs,
        )
    else:
        raise ValueError(arm)

    # Leading space matches normal continuation tokenization after "Label:".
    result = score_options(
        model, tokenizer, blocks, prompt, (" Q"," K"), edit=edit
    )
    labels = ("Q","K")
    p_by_label = {
        label: float(result["conditional_probs"][i])
        for i, label in enumerate(labels)
    }
    pred = str(result["predicted"]).strip()
    return {
        "p_correct": p_by_label[episode.correct_label],
        "predicted_label": pred,
        "correct": pred == episode.correct_label,
        "label_mass": float(result["option_mass"]),
        "format_ok": bool(result["format_ok"]),
        "single_token_fast_path": bool(result["single_token_fast_path"]),
        "option_logprobs": result["logprobs"],
    }

@torch.no_grad()
def score_semantic(model, tokenizer, blocks, episode, vector, layer, factor):
    """Query-only transient state; no demo edits."""
    prompt, ids, positions = prepare_episode(
        tokenizer, episode, model.device, readout="semantic"
    )
    edit = PublishedActAdd(
        layer=layer, vector=vector, factor=factor,
        positions=(positions[-1],), signs=(episode.query_sign,),
    )
    opts = tuple(f" {i}" for i in range(10))
    result = score_options(
        model, tokenizer, blocks, prompt, opts, edit=edit
    )
    probs = result["conditional_probs"]
    expected = float(sum(i * float(probs[i]) for i in range(10)))
    return {
        "semantic_rating": expected,
        "digit_probs": [float(x) for x in probs],
        "semantic_single_token_fast_path": bool(result["single_token_fast_path"]),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--vector-file", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--factor", type=float, default=2.0)
    p.add_argument("--persona", choices=("neutral","upbeat","downbeat"), default="neutral")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-carriers", type=int, default=2)
    a = p.parse_args()

    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.requires_grad_(False)
    blocks = resolve_blocks(model)

    vector = load_hf_vector(
        repo_id=HF_VECTOR_REPO,
        filename=a.vector_file,
        layer=a.layer,
        position=0,
    )
    if vector.numel() != model.config.hidden_size:
        raise SystemExit("vector width does not match model")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name("." + a.out.name + ".tmp")
    with tmp.open("w") as f:
        for carrier in DEV_CARRIERS[:a.max_carriers]:
            episodes = exact_episodes(carrier, a.persona)
            for ep in episodes:
                row = {
                    "model": a.model,
                    "vector_file": a.vector_file,
                    "layer": a.layer,
                    "factor": a.factor,
                    "carrier": carrier,
                    "persona": a.persona,
                    "demo_signs": ep.demo_signs,
                    "query_sign": ep.query_sign,
                    "mapping_id": ep.mapping_id,
                    "correct_label": ep.correct_label,
                    "arms": {},
                }
                for arm in ("clean","query_only","target"):
                    row["arms"][arm] = score_codebook(
                        model, tok, blocks, ep, vector, a.layer, a.factor, arm
                    )
                row.update(score_semantic(
                    model, tok, blocks, ep, vector, a.layer, a.factor
                ))
                f.write(json.dumps(row) + "\n")
                f.flush()
    tmp.replace(a.out)
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
