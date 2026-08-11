#!/usr/bin/env python3
"""Is the steered state present at the position the answer is read from?

Without this, a chance-level codebook report has two explanations and no way to
choose: the hidden state never reached the readout, or it reached it and the
model could not bind it to an arbitrary label. The first is an instrument
failure, the second is a real negative result about introspective access, and
they are worth entirely different amounts.

The in-context semantic channel cannot settle it — measured at 8.99 out of 9 at
*every* span width, so it is at ceiling and would report ~0 whatever the state
did. This reads the residual instead, at the exact position and layer the LM head
consumes, and asks whether the query sign is linearly decodable from it. Decoding
is leave-one-out with a mean-difference direction, so nothing is fitted and
evaluated on the same episode.

* decodable and the codebook report is at chance -> the information is at the
  readout site and goes unused: a real, scoped null about arbitrary-label
  self-report;
* not decodable -> the span edit still does not survive to the answer in this
  prompt, and the null stays uninterpretable.

    python answer_site_probe.py --span-width 49
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from protocol_core import exact_episodes
from rendering import prepare_episode
from welfare_intervention import PublishedActAdd, targeted_published_actadd, load_hf_vector
from run_dev import load_model, expand_spans, DEV_CARRIERS, HF_VECTOR_REPO

@torch.no_grad()
def answer_state(model, tok, blocks, episode, vector, layer, factor, width) -> torch.Tensor:
    """Final-layer residual at the last position — what the LM head actually sees."""
    prompt, ids, positions = prepare_episode(tok, episode, model.device, readout="codebook")
    pos, signs = expand_spans(positions, episode.state_signs, width)
    edit = PublishedActAdd(layer=layer, vector=vector, factor=factor,
                           positions=pos, signs=signs)
    with targeted_published_actadd(blocks, edit):
        out = model(ids, output_hidden_states=True)
    return out.hidden_states[-1][0, -1].float().cpu()

def remove_direction(states: list[torch.Tensor], vector: torch.Tensor) -> list[torch.Tensor]:
    """Strip the injected direction itself out of every state.

    The honest worry about this whole probe: `vGOLD` is *added* at L22 and the
    residual stream carries it forward additively, so a decoder at the final
    layer may be reading the leftover injection rather than anything the model
    computed from it. Passive residue and functional representation predict the
    same decode accuracy. They do not predict the same accuracy once the
    injected direction is projected out.
    """
    unit = (vector / vector.norm()).to(states[0].dtype)
    return [s - (s @ unit) * unit for s in states]

def leave_one_out_decode(states: list[torch.Tensor], labels: list[int]) -> float:
    """Accuracy of a mean-difference direction fitted without the test episode."""
    correct = 0
    for i in range(len(states)):
        rest = [(s, y) for j, (s, y) in enumerate(zip(states, labels, strict=True)) if j != i]
        pos = torch.stack([s for s, y in rest if y == +1]).mean(0)
        neg = torch.stack([s for s, y in rest if y == -1]).mean(0)
        direction = pos - neg
        midpoint = (pos + neg) / 2
        score = float((states[i] - midpoint) @ direction)
        correct += (score > 0) == (labels[i] == +1)
    return correct / len(states)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vector-file", default="concept_vectors/qwen3-4b_step400/goal/mean_diff.pt")
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--factor", type=float, default=2.0)
    p.add_argument("--span-width", type=int, default=49)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    out = a.out or Path(f"results/answer_site_probe_L{a.layer}_w{a.span_width}.json")
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")

    model, tok, blocks = load_model(a.model)
    vector = load_hf_vector(repo_id=HF_VECTOR_REPO, filename=a.vector_file,
                            layer=a.layer, position=0)

    states, signs, carriers = [], [], []
    for carrier in DEV_CARRIERS:
        for ep in exact_episodes(carrier, "neutral"):
            states.append(answer_state(model, tok, blocks, ep, vector,
                                       a.layer, a.factor, a.span_width))
            signs.append(ep.query_sign)
            carriers.append(carrier)
    print(f"{len(states)} episodes, d={states[0].numel()}")

    accuracy = leave_one_out_decode(states, signs)
    stacked = torch.stack(states)
    pos = stacked[[i for i, s in enumerate(signs) if s == +1]].mean(0)
    neg = stacked[[i for i, s in enumerate(signs) if s == -1]].mean(0)
    between = float((pos - neg).norm())
    within = float(torch.stack([s - (pos if y == +1 else neg)
                                for s, y in zip(states, signs, strict=True)]).norm(dim=-1).mean())

    # The same decode restricted to one carrier, so a carrier confound cannot
    # masquerade as sign information.
    per_carrier = {}
    for carrier in DEV_CARRIERS:
        idx = [i for i, c in enumerate(carriers) if c == carrier]
        per_carrier[carrier] = leave_one_out_decode([states[i] for i in idx],
                                                    [signs[i] for i in idx])

    # Leave-one-out folds share training data, so the binomial null is wrong.
    # Permuting the labels through the identical decoder gives the exact null.
    rng = random.Random(0)
    null = []
    for _ in range(2000):
        shuffled = signs[:]
        rng.shuffle(shuffled)
        null.append(leave_one_out_decode(states, shuffled))
    p_value = (1 + sum(x >= accuracy for x in null)) / (1 + len(null))

    # Crux control: is the decode reading a computed state, or the injection?
    stripped = remove_direction(states, vector)
    accuracy_stripped = leave_one_out_decode(stripped, signs)
    pos_s = torch.stack([s for s, y in zip(stripped, signs, strict=True) if y == +1]).mean(0)
    neg_s = torch.stack([s for s, y in zip(stripped, signs, strict=True) if y == -1]).mean(0)
    null_stripped = []
    for _ in range(2000):
        shuffled = signs[:]
        rng.shuffle(shuffled)
        null_stripped.append(leave_one_out_decode(stripped, shuffled))
    p_stripped = (1 + sum(x >= accuracy_stripped for x in null_stripped)) / (1 + len(null_stripped))
    unit = vector / vector.norm()
    cosine = float((pos - neg) @ unit / (pos - neg).norm())

    result = {
        "decode_accuracy_vgold_removed": accuracy_stripped,
        "permutation_p_value_vgold_removed": p_stripped,
        "cosine_decoder_direction_with_vgold": cosine,
        "permutation_p_value": p_value,
        "permutation_null_mean": sum(null) / len(null),
        "permutation_null_p95": sorted(null)[int(0.95 * len(null))],
        "model": a.model, "layer": a.layer, "factor": a.factor,
        "span_width": a.span_width, "n_episodes": len(states),
        "readout": "final-layer hidden state at the last prompt position",
        "decoder": "leave-one-out mean-difference direction, nothing fitted on the test episode",
        "query_sign_decode_accuracy": accuracy,
        "per_carrier_accuracy": per_carrier,
        "between_class_distance": between,
        "mean_within_class_distance": within,
        "separation_ratio": between / within if within else None,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    # Judge against the permuted null, not a round number. A fixed 0.90 bar
    # called p=0.0005 a failure on the first run of this script.
    survives = p_stripped < 0.01 and accuracy_stripped > result["permutation_null_p95"]
    print(f"raw decode {accuracy:.3f} (p={p_value:.4g}); "
          f"with vGOLD projected out {accuracy_stripped:.3f} (p={p_stripped:.4g}); "
          f"cosine with vGOLD {cosine:.3f}")
    print("survives removal of the injected direction — a computed state, not residue"
          if survives else
          "collapses once the injected direction is removed — the decode was reading "
          "the injection's own residue, so R20's claim does not stand")

if __name__ == "__main__":
    main()
