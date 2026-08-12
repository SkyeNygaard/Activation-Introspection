"""What can the four-shot interface learn when nothing is hidden?

Pre-run note: ``notes/16-visible-rule-capacity.md``, written before this ran.

Prompting only. No interventions, no state capture, no site, no training. Tests
whether the four-demonstration Q/K interface can induce a rule and apply it to an
*unseen* instance, or only match a query to something it was already shown.

A ceiling condition (query shown in the demonstrations) and a floor condition
(arbitrary sets, novel query, so no rule exists) bracket every real rule.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Callable

import torch

from introspect import models
from introspect.preflight import check as preflight_check

MODEL = "qwen-3b"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LABELS = ("Q", "K")
ANSWER_PREFIX = "Label:"

Fold = tuple[list[str], list[str], str, str]  # class_a, class_b, query_a, query_b


def _fold(a: list[str], b: list[str], seen: bool) -> Fold:
    """Four demonstrations come from a[:2] and b[:2]; the query is seen or novel."""
    return (a[:2], b[:2], a[0] if seen else a[2], b[0] if seen else b[2])


def _lexical(seen: bool) -> list[Fold]:
    sets = [
        (["velvet", "harbor", "lantern"], ["marble", "cistern", "gantry"]),
        (["thicket", "bellows", "trellis"], ["quarry", "spindle", "cornice"]),
        (["pewter", "furrow", "brambles"], ["gasket", "turret", "lintel"]),
        (["mantle", "shingle", "coppice"], ["ratchet", "flagon", "parapet"]),
    ]
    return [_fold(list(a), list(b), seen) for a, b in sets]


RULES: dict[str, Callable[[], list[Fold]]] = {
    # Ceiling: the query word is one the demonstrations already labelled.
    "lexical_seen": lambda: _lexical(True),
    # Floor: arbitrary sets, novel query. No rule exists to be induced.
    "lexical_unseen": lambda: _lexical(False),
    "first_letter": lambda: [
        _fold(["apple", "otter", "igloo"], ["stone", "candle", "gravel"], False),
        _fold(["eagle", "attic", "ember"], ["timber", "kettle", "pillar"], False),
        _fold(["ocean", "ivory", "anchor"], ["marble", "ladder", "burrow"], False),
        _fold(["urchin", "orbit", "amber"], ["ribbon", "saddle", "nutmeg"], False),
    ],
    "category": lambda: [
        _fold(["otter", "falcon", "badger"], ["hammer", "chisel", "wrench"], False),
        _fold(["salmon", "magpie", "ferret"], ["pliers", "trowel", "mallet"], False),
        _fold(["heron", "weasel", "lizard"], ["spanner", "scalpel", "auger"], False),
        _fold(["sparrow", "beaver", "hornet"], ["drill", "rasp", "clamp"], False),
    ],
    "magnitude": lambda: [
        _fold(["8", "9", "7"], ["2", "1", "3"], False),
        _fold(["7", "6", "9"], ["3", "4", "1"], False),
        _fold(["9", "8", "6"], ["1", "2", "4"], False),
        _fold(["6", "7", "8"], ["4", "3", "2"], False),
    ],
    "parity": lambda: [
        _fold(["4+4", "2+2", "6+2"], ["3+4", "5+2", "1+2"], False),
        _fold(["3+3", "1+1", "5+1"], ["2+3", "4+1", "6+1"], False),
        _fold(["8-2", "4-2", "9-1"], ["7-2", "5-2", "8-1"], False),
        _fold(["5+1", "3+3", "7+1"], ["4+3", "2+1", "6+3"], False),
    ],
}


def render(
    demos: list[tuple[str, str]], query: str, rule_is_arithmetic: bool
) -> str:
    noun = "expression" if rule_is_arithmetic else "item"
    lines = [
        f"Each {noun} below has been given the label Q or K by a fixed hidden rule.",
        "Use the demonstrations, then label the held-out query with one letter.",
    ]
    for item, label in demos:
        lines.extend(["", "Demonstration:", f"Item: {item}", f"Label: {label}"])
    lines.extend(["", "Held-out query:", f"Item: {query}"])
    return "\n".join(lines)


def _single_id(model: models.LoadedModel, prompt: str, label: str) -> int:
    base = model.encode(prompt)
    cont = model.encode(prompt + label)
    if int(cont.shape[1]) != int(base.shape[1]) + 1:
        raise ValueError(f"label {label!r} is not one token here")
    return int(cont[0, -1])


@torch.no_grad()
def score(model: models.LoadedModel, prompt: str, correct: str) -> dict[str, object]:
    ids = tuple(_single_id(model, prompt, f" {label}") for label in LABELS)
    logits = model.forward_logits(model.encode(prompt))[0, -1].float()
    selected = logits[torch.tensor(ids, device=logits.device)]
    predicted = LABELS[int(selected.argmax())]
    return {
        "predicted": predicted,
        "correct": predicted == correct,
        "format_ok": int(logits.argmax()) in set(ids),
        "label_mass": float(
            torch.logsumexp(selected, 0).sub(torch.logsumexp(logits, 0)).exp()
        ),
    }


def cells() -> list[tuple[tuple[int, ...], int, int]]:
    """Six balanced demonstration orders x two label maps x two query classes."""
    orders = sorted(set(itertools.permutations((0, 0, 1, 1))))
    return [(o, m, q) for o in orders for m in (0, 1) for q in (0, 1)]


def twin_pair(rows: list[dict[str, object]]) -> float:
    groups: dict[tuple[str, int, tuple, int], list[bool]] = {}
    for r in rows:
        key = (str(r["rule"]), int(r["fold"]), tuple(r["order"]), int(r["map"]))  # type: ignore[arg-type]
        groups.setdefault(key, []).append(bool(r["correct"]))
    full = [v for v in groups.values() if len(v) == 2]
    return sum(all(v) for v in full) / len(full) if full else float("nan")


def _self_check() -> None:
    assert len(cells()) == 24, "the design is 6 orders x 2 maps x 2 query classes"
    a, b, qa, qb = _fold(["x", "y", "z"], ["p", "q", "r"], seen=True)
    assert qa == "x" and qa in a, "seen query must appear in the demonstrations"
    a, b, qa, qb = _fold(["x", "y", "z"], ["p", "q", "r"], seen=False)
    assert qa == "z" and qa not in a, "unseen query must not appear in them"


def run(args: argparse.Namespace) -> None:
    _self_check()
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

        design = cells()[:2] if args.smoke else cells()
        rows: list[dict[str, object]] = []
        for rule, build in RULES.items():
            folds = build()[:1] if args.smoke else build()
            arithmetic = rule in {"parity", "magnitude"}
            for fold_id, (class_a, class_b, query_a, query_b) in enumerate(folds):
                for order, map_id, query_class in design:
                    label_a = LABELS[map_id]
                    label_b = LABELS[1 - map_id]
                    pool = {0: list(class_a), 1: list(class_b)}
                    used = {0: 0, 1: 0}
                    demos = []
                    for which in order:
                        demos.append(
                            (pool[which][used[which]], label_a if which == 0 else label_b)
                        )
                        used[which] += 1
                    query = query_a if query_class == 0 else query_b
                    correct = label_a if query_class == 0 else label_b
                    prompt = model.chat(render(demos, query, arithmetic), ANSWER_PREFIX)
                    rows.append(
                        {
                            "rule": rule,
                            "fold": fold_id,
                            "order": list(order),
                            "map": map_id,
                            "query_class": query_class,
                            "query": query,
                            **score(model, prompt, correct),
                        }
                    )
            print(f"{rule}: done", flush=True)

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

        by_rule = {}
        for rule in RULES:
            subset = [r for r in rows if r["rule"] == rule]
            by_rule[rule] = {
                "accuracy": sum(bool(r["correct"]) for r in subset) / len(subset),
                "twin_pair": twin_pair(subset),
                "format_rate": sum(bool(r["format_ok"]) for r in subset) / len(subset),
                "n": len(subset),
            }
        summary = {
            "what_this_is": (
                "Prompting-only screen: can the four-shot Q/K interface induce a rule "
                "and apply it to an unseen instance, or only match a shown query?"
            ),
            "smoke": bool(args.smoke),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "ceiling": "lexical_seen",
            "floor": "lexical_unseen",
            "by_rule": by_rule,
            "elapsed_seconds": round(time.time() - started, 1),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"wrote {out} and {summary_path}", flush=True)
    finally:
        model.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=Path("results/visible_rule_screen_v1_raw.jsonl")
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
