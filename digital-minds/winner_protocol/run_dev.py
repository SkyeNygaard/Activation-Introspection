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

def load_model(name: str):
    """Load one model onto the best available local device.

    The device is named rather than left to `device_map="auto"`: on this 24 GB
    Mac a plan that stages weights through CPU, or a `.to("mps")` after a CPU
    load, doubles peak memory and pushes a 4B run into swap.
    """
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=torch.bfloat16,
        device_map="mps" if torch.backends.mps.is_available() else "auto",
        trust_remote_code=True,
    )
    model.eval()
    model.requires_grad_(False)
    return model, tok, resolve_blocks(model)

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

def expand_spans(positions, signs, width: int):
    """Widen each state from its marker token to the `width` tokens ending on it.

    R18 measured the marker token itself to be causally inert, and
    `span_threshold.py` measured a distal 51-token window to carry +1.65 of the
    3.96 whole-prompt effect. Width is therefore the protocol's sensitivity knob,
    not a free parameter: at width 1 this is exactly the original edit.
    """
    out_positions, out_signs = [], []
    for pos, sign in zip(positions, signs, strict=True):
        span = range(max(0, pos - width + 1), pos + 1)
        out_positions.extend(span)
        out_signs.extend([sign] * len(span))
    if len(set(out_positions)) != len(out_positions):
        raise ValueError(f"state spans overlap at width {width}; markers are too close")
    return tuple(out_positions), tuple(out_signs)

@torch.no_grad()
def score_codebook(model, tokenizer, blocks, episode, vector, layer, factor, arm, width=1):
    prompt, ids, positions = prepare_episode(
        tokenizer, episode, model.device, readout="codebook"
    )

    if arm == "clean":
        edit = None
    elif arm == "query_only":
        pos, signs = expand_spans((positions[-1],), (episode.query_sign,), width)
        edit = PublishedActAdd(
            layer=layer, vector=vector, factor=factor, positions=pos, signs=signs,
        )
    elif arm == "target":
        pos, signs = expand_spans(positions, episode.state_signs, width)
        edit = PublishedActAdd(
            layer=layer, vector=vector, factor=factor, positions=pos, signs=signs,
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
def score_semantic(model, tokenizer, blocks, episode, vector, layer, factor, width=1):
    """Query-only transient state; no demo edits."""
    prompt, ids, positions = prepare_episode(
        tokenizer, episode, model.device, readout="semantic"
    )
    pos, signs = expand_spans((positions[-1],), (episode.query_sign,), width)
    edit = PublishedActAdd(
        layer=layer, vector=vector, factor=factor, positions=pos, signs=signs,
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
    p.add_argument("--span-width", type=int, default=1,
                   help="tokens per hidden state, ending on its marker; 1 is the "
                        "original marker-only edit that R18 showed to be inert")
    a = p.parse_args()

    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    model, tok, blocks = load_model(a.model)

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
                    "span_width": a.span_width,
                    "arms": {},
                }
                for arm in ("clean","query_only","target"):
                    row["arms"][arm] = score_codebook(
                        model, tok, blocks, ep, vector, a.layer, a.factor, arm,
                        a.span_width
                    )
                row.update(score_semantic(
                    model, tok, blocks, ep, vector, a.layer, a.factor, a.span_width
                ))
                f.write(json.dumps(row) + "\n")
                f.flush()
    tmp.replace(a.out)
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
