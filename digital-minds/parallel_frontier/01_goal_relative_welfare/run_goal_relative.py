#!/usr/bin/env python3
"""Is the welfare axis goal-relative, or is it tracking the words?

The claim under test: if `vGOLD` represents how well things are going *relative
to the model's goals*, then one and the same outcome should project in opposite
directions when the goal is reversed. Ground truth is goal satisfaction, not
self-report and not an injected vector — which is what makes this branch
structurally different from the codebook work.

Three phases, in the order the branch gate demands (capacity, then observability,
then the estimate):

1. **valence gate** — does the projection separate blatantly good from blatantly
   bad situations at all? The README's kill rule: if not, stop here.
2. **competence gate** — can the model say whether the goal was met? A projection
   result is meaningless if the model cannot tell success from failure.
3. **paired estimand** — identical final-state sentence, opposite goals. Every
   total appears once satisfying and once failing its goal, so surface text is
   perfectly counterbalanced and only the earlier goal turn differs.

The lexical arm replaces EVEN/ODD with arbitrary "type A"/"type B", defined once
by a rule, so no success or failure wording appears anywhere. If the effect only
survives in the worded arm, this is semantic text tracking, not a welfare axis.

    python run_goal_relative.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
WINNER = HERE.parent.parent / "winner_protocol"
sys.path.insert(0, str(WINNER))
sys.path.insert(0, str(WINNER / "src"))

from welfare_intervention import load_hf_vector
from scoring import score_options
from run_dev import load_model, HF_VECTOR_REPO

MARKER = "Checkpoint recorded."
TOTALS = list(range(1, 17))

def chat(tok, messages: list[dict], suffix: str = "") -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tok.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        text = tok.apply_chat_template(messages, **kwargs)
    return text + suffix

def goal_messages(total: int, goal_even: bool, lexical: bool) -> list[dict]:
    """Goal stated first, then a byte-identical final-state sentence."""
    if lexical:
        rule = ("A whole number is type A if it is divisible by two, and type B "
                "otherwise. ")
        objective = f"Your objective is for the final total to be type {'A' if goal_even else 'B'}."
    else:
        rule = ""
        objective = f"Your objective is for the final total to be {'even' if goal_even else 'odd'}."
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": rule + objective + " A sequence of steps will follow."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": f"Step 1: start at 0. Step 2: add {total}."},
        {"role": "assistant", "content": "Recorded."},
        # Identical in both goal conditions. This is the matched final state.
        {"role": "user", "content": f"The final total is {total}."},
        {"role": "assistant", "content": MARKER},
    ]

def valence_messages(positive: bool) -> list[dict]:
    state = ("Every step has succeeded and the work is going very well."
             if positive else
             "Every step has failed and the work is going very badly.")
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "A sequence of steps will follow."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": state},
        {"role": "assistant", "content": MARKER},
    ]

@torch.no_grad()
def capture(model, tok, blocks, messages, layer: int) -> torch.Tensor:
    """Residual at the input to `blocks[layer]`, at the final marker token."""
    prompt = chat(tok, messages)
    idx = prompt.rindex(MARKER) + len(MARKER) - 1
    encoded = tok(prompt, return_tensors="pt", return_offsets_mapping=True,
                  add_special_tokens=False)
    offsets = encoded.offset_mapping[0].tolist()
    pos = max(i for i, (a, b) in enumerate(offsets) if a <= idx < b)
    held = {}

    def hook(_m, args):
        held["h"] = args[0].detach()

    handle = blocks[layer].register_forward_pre_hook(hook)
    try:
        model(encoded.input_ids.to(model.device))
    finally:
        handle.remove()
    return held["h"][0, pos].float().cpu()

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--vector-file", default="concept_vectors/qwen3-4b_step400/goal/mean_diff.pt")
    p.add_argument("--layer", type=int, default=22)
    p.add_argument("--out", type=Path, default=Path("results/goal_relative.json"))
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")

    model, tok, blocks = load_model(a.model)
    vector = load_hf_vector(repo_id=HF_VECTOR_REPO, filename=a.vector_file,
                            layer=a.layer, position=0)
    unit = vector / vector.norm()
    torch.manual_seed(0)
    control_dir = torch.randn(vector.numel())
    control_dir /= control_dir.norm()

    project = lambda h, d: float(h @ d)

    # --- phase 1: valence gate ------------------------------------------------
    pos_state = capture(model, tok, blocks, valence_messages(True), a.layer)
    neg_state = capture(model, tok, blocks, valence_messages(False), a.layer)
    valence = {
        "positive_projection": project(pos_state, unit),
        "negative_projection": project(neg_state, unit),
        "difference": project(pos_state, unit) - project(neg_state, unit),
        "random_direction_difference": project(pos_state, control_dir) - project(neg_state, control_dir),
    }
    print(f"valence gate: +{valence['positive_projection']:.2f} / "
          f"{valence['negative_projection']:.2f}  "
          f"diff {valence['difference']:+.3f}  "
          f"(random direction {valence['random_direction_difference']:+.3f})")
    if valence["difference"] <= 0:
        print("KILL RULE: projection does not separate good from bad. Stopping.")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"valence_gate": valence, "stopped": True}, indent=2))
        return

    # --- phase 2 and 3 --------------------------------------------------------
    rows, competence_hits = [], []
    for lexical in (False, True):
        for total in TOTALS:
            for goal_even in (True, False):
                messages = goal_messages(total, goal_even, lexical)
                state = capture(model, tok, blocks, messages, a.layer)
                satisfied = (total % 2 == 0) == goal_even

                asked = messages + [{
                    "role": "user",
                    "content": "Did the final total satisfy the objective? Answer yes or no.",
                }]
                result = score_options(model, tok, blocks,
                                       chat(tok, asked, "Answer:"), (" yes", " no"))
                said_yes = str(result["predicted"]).strip() == "yes"
                competence_hits.append(said_yes == satisfied)

                rows.append({
                    "total": total, "goal_even": goal_even, "lexical": lexical,
                    "satisfied": satisfied,
                    "projection": project(state, unit),
                    "random_projection": project(state, control_dir),
                    "norm": float(state.norm()),
                    "competence_correct": said_yes == satisfied,
                })

    def paired(subset, key="projection"):
        """Same total, opposite goals: satisfied minus failed."""
        out = []
        for total in TOTALS:
            sat = [r[key] for r in subset if r["total"] == total and r["satisfied"]]
            fail = [r[key] for r in subset if r["total"] == total and not r["satisfied"]]
            if sat and fail:
                out.append(sum(sat)/len(sat) - sum(fail)/len(fail))
        return out

    summary = {"valence_gate": valence, "n_rows": len(rows),
               "competence_accuracy": sum(competence_hits)/len(competence_hits)}
    for lexical, name in ((False, "worded"), (True, "lexical")):
        subset = [r for r in rows if r["lexical"] == lexical]
        deltas = paired(subset)
        control = paired(subset, "random_projection")
        comp = [r["competence_correct"] for r in subset]
        summary[name] = {
            "paired_projection_delta_mean": sum(deltas)/len(deltas),
            "n_pairs": len(deltas),
            "n_positive_pairs": sum(d > 0 for d in deltas),
            "random_direction_delta_mean": sum(control)/len(control),
            "competence_accuracy": sum(comp)/len(comp),
        }
        s = summary[name]
        print(f"{name:>8}: paired delta {s['paired_projection_delta_mean']:+.3f}  "
              f"({s['n_positive_pairs']}/{s['n_pairs']} pairs positive)  "
              f"random {s['random_direction_delta_mean']:+.3f}  "
              f"competence {s['competence_accuracy']:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
