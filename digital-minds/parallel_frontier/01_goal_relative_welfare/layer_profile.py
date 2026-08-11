#!/usr/bin/env python3
"""Does the goal-relative signal live where vGOLD is causally active?

`run_goal_relative.py` found the same outcome projecting higher on `vGOLD` when
it satisfies the goal than when it fails it — 16/16 matched pairs, in both the
worded and the wordless lexical arm. A random direction shows nothing, which
rules out "any direction would do this", but not the more interesting objection:
that *any semantically meaningful* direction would, and `vGOLD` is incidental.

The independent handle is causal. `winner_protocol/localize_source_layer.jsonl`
measured where steering along `vGOLD` actually moves behaviour: unimodal, peaking
at L22, inert from L25. If the goal-relative *representation* tracks that profile
— present where the direction is causally load-bearing, absent where it is not —
the two measurements are about the same thing. If the projection effect is flat
across depth, it is a generic property of the readout and the causal localization
is unrelated.

Every layer's residual comes from one forward pass, so this costs the same as the
single-layer run.

    python layer_profile.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
WINNER = HERE.parent.parent / "winner_protocol"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WINNER))
sys.path.insert(0, str(WINNER / "src"))

from welfare_intervention import load_hf_vector
from run_dev import load_model, HF_VECTOR_REPO
from run_goal_relative import goal_messages, chat, MARKER, TOTALS

@torch.no_grad()
def capture_all_layers(model, tok, blocks, messages) -> torch.Tensor:
    """[n_layers, d] residual at each block input, at the final marker token."""
    prompt = chat(tok, messages)
    idx = prompt.rindex(MARKER) + len(MARKER) - 1
    encoded = tok(prompt, return_tensors="pt", return_offsets_mapping=True,
                  add_special_tokens=False)
    offsets = encoded.offset_mapping[0].tolist()
    pos = max(i for i, (a, b) in enumerate(offsets) if a <= idx < b)

    held: dict[int, torch.Tensor] = {}
    handles = [
        block.register_forward_pre_hook(
            lambda _m, args, i=i: held.__setitem__(i, args[0][0, pos].detach().float().cpu())
        )
        for i, block in enumerate(blocks)
    ]
    try:
        model(encoded.input_ids.to(model.device))
    finally:
        for handle in handles:
            handle.remove()
    return torch.stack([held[i] for i in range(len(blocks))])

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vector-file", default="concept_vectors/qwen3-4b_step400/goal/mean_diff.pt")
    p.add_argument("--out", type=Path, default=Path("results/goal_relative_layer_profile.json"))
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    model, tok, blocks = load_model(a.model)
    n_layers = len(blocks)
    units = torch.stack([
        (lambda v: v / v.norm())(load_hf_vector(repo_id=HF_VECTOR_REPO,
                                                filename=a.vector_file,
                                                layer=layer, position=0))
        for layer in range(n_layers)
    ])
    torch.manual_seed(0)
    control = torch.randn(n_layers, units.shape[1])
    control /= control.norm(dim=-1, keepdim=True)

    states: dict[tuple, torch.Tensor] = {}
    for lexical in (False, True):
        for total in TOTALS:
            for goal_even in (True, False):
                states[(lexical, total, goal_even)] = capture_all_layers(
                    model, tok, blocks, goal_messages(total, goal_even, lexical))
    print(f"captured {len(states)} states x {n_layers} layers")

    # Causal steering profile from the winner_protocol localization, for comparison.
    causal_path = WINNER / "results" / "localize_source_layer.jsonl"
    causal = {}
    if causal_path.exists():
        for line in causal_path.read_text().splitlines():
            row = json.loads(line)
            if row["factor"] == 2.0:
                causal[row["layer"]] = row["plus_minus"]

    print(f"{'layer':>5} {'worded':>9} {'lexical':>9} {'random':>9} {'causal':>9}")
    rows = []
    for layer in range(n_layers):
        entry = {"layer": layer, "causal_steering_delta_f2": causal.get(layer)}
        for lexical, name in ((False, "worded"), (True, "lexical")):
            deltas, control_deltas = [], []
            for total in TOTALS:
                satisfied_goal = (total % 2 == 0)
                sat = states[(lexical, total, satisfied_goal)][layer]
                fail = states[(lexical, total, not satisfied_goal)][layer]
                deltas.append(float((sat - fail) @ units[layer]))
                control_deltas.append(float((sat - fail) @ control[layer]))
            entry[name] = sum(deltas) / len(deltas)
            entry[f"{name}_pairs_positive"] = sum(d > 0 for d in deltas)
            entry[f"{name}_random"] = sum(control_deltas) / len(control_deltas)
        rows.append(entry)
        print(f"{layer:>5} {entry['worded']:>+9.3f} {entry['lexical']:>+9.3f} "
              f"{entry['worded_random']:>+9.3f} "
              f"{(f'{causal[layer]:+.3f}' if layer in causal else '—'):>9}")

    # A single random direction is a noisy baseline, and the raw deltas grow with
    # depth simply because residual norms do. Compare vGOLD's sign consistency
    # against the distribution over many random directions at the same layer.
    focus = [17, 20, 21, 22, 25, 29, 35]
    permutation = {}
    generator = torch.Generator().manual_seed(1)
    for layer in focus:
        diffs = torch.stack([
            states[(lexical, total, (total % 2 == 0))][layer]
            - states[(lexical, total, not (total % 2 == 0))][layer]
            for lexical in (False, True) for total in TOTALS
        ])
        worded_diffs, lexical_diffs = diffs[:len(TOTALS)], diffs[len(TOTALS):]
        entry = {}
        for name, block in (("worded", worded_diffs), ("lexical", lexical_diffs)):
            observed = int(((block @ units[layer]) > 0).sum())
            directions = torch.randn(2000, units.shape[1], generator=generator)
            directions /= directions.norm(dim=-1, keepdim=True)
            null = ((block @ directions.T) > 0).sum(dim=0)
            entry[name] = {
                "observed_pairs_positive": observed,
                "null_mean": float(null.float().mean()),
                "null_p95": int(null.float().quantile(0.95)),
                "p_value": float(((null >= observed).sum() + 1) / (len(null) + 1)),
            }
        permutation[layer] = entry
        print(f"L{layer}: worded {entry['worded']['observed_pairs_positive']}/16 "
              f"p={entry['worded']['p_value']:.4f} | "
              f"lexical {entry['lexical']['observed_pairs_positive']}/16 "
              f"p={entry['lexical']['p_value']:.4f} "
              f"(null mean {entry['lexical']['null_mean']:.1f}/16)")

    live = [r for r in rows if r["causal_steering_delta_f2"] is not None
            and r["causal_steering_delta_f2"] >= 1.0]
    dead = [r for r in rows if r["causal_steering_delta_f2"] is not None
            and r["causal_steering_delta_f2"] < 0.25]
    summary = {
        "n_layers": n_layers,
        "causally_live_layers": [r["layer"] for r in live],
        "causally_dead_layers": [r["layer"] for r in dead],
        "mean_worded_delta_live": sum(r["worded"] for r in live) / max(1, len(live)),
        "mean_worded_delta_dead": sum(r["worded"] for r in dead) / max(1, len(dead)),
        "mean_lexical_delta_live": sum(r["lexical"] for r in live) / max(1, len(live)),
        "mean_lexical_delta_dead": sum(r["lexical"] for r in dead) / max(1, len(dead)),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n" + json.dumps(summary, indent=1))
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
