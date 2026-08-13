#!/usr/bin/env python3
"""notes/30 -- can a model tell, before it answers, that it is about to be wrong?

Every result in this repository measures a model reporting on a state someone else
injected. This one does not. The internal state is whatever the model has while
working a hard sum, and the two classes are "gets it right" and "gets it wrong".
Ground truth is arithmetic; nothing is planted and nothing has to clump.

Three predictors of the same binary outcome, notes/20's comparator ladder rebuilt
on a natural state:

``verbal``
    the model is asked "will you get this right?" before it answers
``margin``
    the model's own confidence in the answer it then gives
``probe``
    a logistic probe on the residual state at the last token of the question,
    fitted on a training split

Scored by AUROC on a held-out split, because notes/29 established that ranking is
the right instrument and averaging is what made notes/20 wrong.

Two gates run first and can stop the run:

    A  the task must be in the usable accuracy band, or nothing is predictable
    B  the probe must beat chance, or the state does not carry the information
       and the model's silence says nothing about introspection

    uv run python scripts/run_self_knowledge.py --out results/self_knowledge_v1_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_category_geometry import LAYER, MODEL, MODEL_REVISION

from introspect import models
from introspect.hooks import capture
from introspect.preflight import check as preflight_check

#: Gate A sweeps these and freezes the one whose accuracy is nearest TARGET_ACC.
DIGIT_SIZES = ((2, 1), (2, 2), (3, 2))
TARGET_ACC = 0.65
USABLE_BAND = (0.55, 0.85)
GATE_A_N = 60

#: Gate B: below this the state does not carry impending failure and the run stops.
GATE_B_AUROC = 0.55

N_PROBLEMS = 400

#: notes/30 confirmation. The development run found the model's signal beats
#: problem size only inside the hardest third, and that band was chosen by looking.
#: --hard-only regenerates fresh problems restricted to that regime so the same
#: comparison can be made once, pre-registered, on data never seen.
HARD_MIN_PRODUCT = 2000
CONFIRM_SEED = 4242
TRAIN_FRACTION = 0.5
MAX_NEW = 12
SEED = 0

ANSWER_ASK = "Compute {a} x {b}. Reply with the number only, nothing else."
VERBAL_ASK = (
    "You are about to be asked to compute {a} x {b}.\n"
    "Before you try: will you get it right?\n"
    "Answer with exactly one word, Yes or No."
)


def problems(
    size: tuple[int, int], n: int, seed: int, min_product: int = 0
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    lo_a, hi_a = 10 ** (size[0] - 1), 10 ** size[0] - 1
    lo_b, hi_b = 10 ** (size[1] - 1), 10 ** size[1] - 1
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    while len(out) < n:
        p = (rng.randint(lo_a, hi_a), rng.randint(lo_b, hi_b))
        if p[0] * p[1] < min_product:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(a positive outranks a negative). Ties count half. notes/29's instrument."""
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


@torch.no_grad()
def ask_answer(model: models.LoadedModel, a: int, b: int) -> dict[str, Any]:
    """Generate the answer, and capture the state at the last question token."""
    prompt = model.chat(ANSWER_ASK.format(a=a, b=b))
    ids = model.encode(prompt)
    with capture(model, [LAYER]) as store:
        logits = model.forward_logits(ids)[0, -1].float()
    state = store.acts[LAYER][0][0, -1].float().cpu()

    # Confidence in the answer it is about to give: gap between the top two
    # next-token logits at the first answer position.
    top2 = torch.topk(logits, 2).values
    margin = float(top2[0] - top2[1])

    gen = model.generate_ids(ids, max_new_tokens=MAX_NEW, do_sample=False)
    text = model.tokenizer.decode(gen[0][int(ids.shape[1]) :], skip_special_tokens=True)
    found = re.search(r"-?\d+", text.replace(",", ""))
    return {
        "answer_text": text.strip()[:40],
        "answer": int(found.group()) if found else None,
        "margin": margin,
        "state": state,
    }


@torch.no_grad()
def ask_verbal(model: models.LoadedModel, a: int, b: int) -> dict[str, Any]:
    """Prospective self-prediction, asked before the model has answered."""
    prompt = model.chat(VERBAL_ASK.format(a=a, b=b), assistant_prefix="")
    ids = model.encode(prompt)
    logits = model.forward_logits(ids)[0, -1].float()
    yes = model.tokenizer(" Yes", add_special_tokens=False).input_ids[0]
    no = model.tokenizer(" No", add_special_tokens=False).input_ids[0]
    yes_bare = model.tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_bare = model.tokenizer("No", add_special_tokens=False).input_ids[0]
    y = float(max(logits[yes], logits[yes_bare]))
    n = float(max(logits[no], logits[no_bare]))
    return {"verbal_yes_minus_no": y - n, "verbal_says_yes": bool(y > n)}


def fit_probe(states: list[torch.Tensor], labels: list[bool], epochs: int = 300) -> torch.Tensor:
    """Logistic probe, plain gradient descent. The cheap outsider."""
    x = torch.stack(states)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    x = torch.cat([x, torch.ones(len(x), 1)], dim=1)
    y = torch.tensor([float(v) for v in labels])
    w = torch.zeros(x.shape[1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w, y)
        loss = loss + 1e-3 * (w * w).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    return w.detach()


def probe_scores(
    w: torch.Tensor, states: list[torch.Tensor], ref: list[torch.Tensor]
) -> list[float]:
    r = torch.stack(ref)
    mean, std = r.mean(0), r.std(0) + 1e-6
    x = (torch.stack(states) - mean) / std
    x = torch.cat([x, torch.ones(len(x), 1)], dim=1)
    return (x @ w).tolist()


def gate_a(model: models.LoadedModel) -> tuple[int, int]:
    """Pick the digit size whose accuracy is nearest the target. Frozen after this."""
    best, best_gap, table = None, 1e9, {}
    for size in DIGIT_SIZES:
        hits = 0
        for a, b in problems(size, GATE_A_N, seed=SEED + 99):
            r = ask_answer(model, a, b)
            hits += r["answer"] == a * b
        acc = hits / GATE_A_N
        table[f"{size[0]}x{size[1]}"] = acc
        print(f"  gate A {size[0]}x{size[1]}: accuracy {acc:.3f}", flush=True)
        if USABLE_BAND[0] <= acc <= USABLE_BAND[1] and abs(acc - TARGET_ACC) < best_gap:
            best, best_gap = size, abs(acc - TARGET_ACC)
    if best is None:
        raise SystemExit(
            f"gate A failed: no digit size in the usable band {USABLE_BAND}. "
            f"Accuracies: {table}. Change the task, do not run the sweep."
        )
    print(f"  gate A frozen: {best[0]}x{best[1]}", flush=True)
    return best


def run(args: argparse.Namespace) -> None:
    out = args.out
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    for path in (out, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    started = time.time()
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

        if args.hard_only:
            size, n = (2, 2), N_PROBLEMS
            seed, min_product = CONFIRM_SEED, HARD_MIN_PRODUCT
            print(f"confirmation mode: 2x2, product >= {HARD_MIN_PRODUCT}", flush=True)
        else:
            size = (2, 1) if args.smoke else gate_a(model)
            n = 24 if args.smoke else N_PROBLEMS
            seed, min_product = SEED, 0
        rows: list[dict[str, Any]] = []
        states: list[torch.Tensor] = []

        for i, (a, b) in enumerate(problems(size, n, seed=seed, min_product=min_product)):
            ans = ask_answer(model, a, b)
            verbal = ask_verbal(model, a, b)
            states.append(ans.pop("state"))
            rows.append(
                {
                    "index": i,
                    "a": a,
                    "b": b,
                    "truth": a * b,
                    "correct": ans["answer"] == a * b,
                    **{k: v for k, v in ans.items() if k != "state"},
                    **verbal,
                }
            )
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{n} ({time.time() - started:.0f}s)", flush=True)

        acc = sum(bool(r["correct"]) for r in rows) / len(rows)
        yes_rate = sum(bool(r["verbal_says_yes"]) for r in rows) / len(rows)
        print(f"\naccuracy {acc:.3f}   verbal yes-rate {yes_rate:.3f}", flush=True)

        cut = int(len(rows) * TRAIN_FRACTION)
        idx = list(range(len(rows)))
        random.Random(seed).shuffle(idx)
        train, test = idx[:cut], idx[cut:]
        w = fit_probe([states[i] for i in train], [rows[i]["correct"] for i in train])
        scores = probe_scores(w, [states[i] for i in test], [states[i] for i in train])
        for j, i in enumerate(test):
            rows[i]["probe_score"] = scores[j]

        held = [rows[i] for i in test]
        pos = [r for r in held if r["correct"]]
        neg = [r for r in held if not r["correct"]]
        ladder = {
            # The control notes/30 did not pre-register and should have: a feature
            # needing no access to the model at all. Bigger sums are harder.
            "product_size_alone": auroc(
                [-(r["a"] * r["b"]) for r in pos], [-(r["a"] * r["b"]) for r in neg]
            ),
            "probe": auroc([r["probe_score"] for r in pos], [r["probe_score"] for r in neg]),
            "margin": auroc([r["margin"] for r in pos], [r["margin"] for r in neg]),
            "verbal": auroc(
                [r["verbal_yes_minus_no"] for r in pos],
                [r["verbal_yes_minus_no"] for r in neg],
            ),
        }

        gate_b_pass = ladder["probe"] >= GATE_B_AUROC
        kill = yes_rate > 0.95 or yes_rate < 0.05

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
        summary = {
            "note": "notes/30",
            "model": MODEL,
            "revision": MODEL_REVISION,
            "layer": LAYER,
            "digit_size": f"{size[0]}x{size[1]}",
            "hard_only": bool(args.hard_only),
            "min_product": min_product,
            "n_problems": len(rows),
            "n_test": len(held),
            "task_accuracy": acc,
            "verbal_yes_rate": yes_rate,
            "verbal_is_constant_responder": kill,
            "auroc": ladder,
            "gate_b_probe_threshold": GATE_B_AUROC,
            "gate_b_passed": gate_b_pass,
            "smoke": bool(args.smoke),
            "elapsed_seconds": round(time.time() - started, 1),
            "cost_note": (
                "The probe is fitted on a training split of the model's own errors; "
                "the model is asked cold. This is NOT a cost-matched comparison and "
                "no conclusion may be phrased as one. Stated in notes/30 before the run."
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not gate_b_pass:
            print(
                "\nGATE B FAILED: the state does not carry impending failure. "
                "Report that, and say nothing about introspection.",
                flush=True,
            )
    finally:
        model.free()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--hard-only",
        action="store_true",
        help="notes/30 confirmation: fresh seed, products >= HARD_MIN_PRODUCT",
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
