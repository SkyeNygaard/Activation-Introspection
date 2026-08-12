"""Does hidden-state reporting abstract over a category, or match a prototype?

Pre-run note: ``notes/23-held-out-semantic-generalization.md``, written before
this ran.

``14`` injected ``eagle`` at the demonstration positions and ``eagle`` at the
query, so the model never had to know what a bird is -- only that the query state
sat near the state labelled ``Q``. That is prototype matching, and it produces
every number ``14`` reports.

Here **every injection position gets a different exemplar**. The demonstrations
carry two members of each category; the query carries a member that appears in no
demonstration. Answering requires placing an unseen state into a category defined
by two examples.

Frozen episode machinery is imported, not modified: ``exact_episodes``,
``prepare_episode`` and the twin-pair metric are unchanged, so the visible text is
still byte-identical across query twins and an input-only strategy is still pinned
at chance by construction. The only new logic is which vector goes at which
position.

Primary metric is **twin-pair** accuracy. ``22`` is the reason: at the row level a
model that ignores the state and repeats one label reads as 0.497, which looks
exactly like chance.

Requires the capacity gate to have run and passed:

    uv run python scripts/run_category_geometry.py
    uv run python scripts/run_heldout_semantic.py --out results/heldout_semantic_v1_raw.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from run_category_geometry import CANDIDATES, LAYER, MODEL, MODEL_REVISION
from run_category_geometry import OUT as GATE_OUT
from run_content_vs_disturbance import matched_strength, twin_pair
from run_matched_reader import _read

from introspect import models
from introspect.codebook_icl import (
    CONFIRM_VISIBLE_SAMPLES,
    LABELS,
    exact_episodes,
    prepare_episode,
)
from introspect.concepts import ConceptVector, build_concept_vector, random_control
from introspect.hooks import Intervention, capture, intervene
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

READERS = ("centroid_euclidean", "centroid_cosine", "shuffled_labels")
ARMS = (
    "same_exemplar",
    "heldout_semantic",
    "heldout_scrambled",
    "heldout_random",
    "query_only",
)

#: Independent exemplar assignments per category pair. Each redraws which two
#: exemplars demonstrate and which held-out one is queried, so a result cannot be
#: an accident of one lucky exemplar.
DRAWS = 3

#: Opposite poles give 1.0 and orthogonal directions about 1.41. Beyond this the
#: two classes were nearly collinear and matching has stopped being a control.
MAX_MATCHED_STRENGTH = 6.0


@torch.no_grad()
def build_centered_bank(model: models.LoadedModel, names: list[str]) -> dict[str, ConceptVector]:
    """Concept vectors centred on an independent bank.

    Centring on the evaluation bank's own mean would push the two categories to
    opposite sides by construction, manufacturing the structure under test.
    """
    center = torch.stack(
        [build_concept_vector(model, n, LAYER).vector for n in CENTERING_CONCEPTS]
    ).mean(0)
    return {
        n: ConceptVector(
            name=n, layer=LAYER, vector=build_concept_vector(model, n, LAYER).vector - center
        )
        for n in names
    }


def assign(
    a_names: list[str], b_names: list[str], *, draw: int, same_exemplar: bool
) -> dict[str, Any]:
    """Pick demonstration exemplars and a query exemplar held out from them."""
    rng = random.Random(draw)
    demo_a = rng.sample(a_names, 2)
    demo_b = rng.sample(b_names, 2)
    if same_exemplar:
        # notes/14's design: one vector per class, reused at the query.
        demo_a = [demo_a[0], demo_a[0]]
        demo_b = [demo_b[0], demo_b[0]]
        return {"demo_a": demo_a, "demo_b": demo_b, "query_a": demo_a[0], "query_b": demo_b[0]}
    return {
        "demo_a": demo_a,
        "demo_b": demo_b,
        "query_a": rng.choice([n for n in a_names if n not in demo_a]),
        "query_b": rng.choice([n for n in b_names if n not in demo_b]),
    }


def exemplar_interventions(
    bank: dict[str, ConceptVector],
    plan: dict[str, Any],
    positions: tuple[int, ...],
    signs: tuple[int, ...],
    *,
    strength: float,
) -> list[Intervention]:
    """One edit per position, each carrying its own exemplar.

    ``pair_interventions`` in ``run_content_vs_disturbance`` gives every position
    of a sign the same vector; that is exactly the prototype confound. Here the
    two demonstrations of a class carry different exemplars and the query carries
    a third.
    """
    if len(positions) != len(signs):
        raise ValueError("each state sign needs one intervention position")
    seen = {1: 0, -1: 0}
    out = []
    for idx, (position, sign) in enumerate(zip(positions, signs, strict=True)):
        is_query = idx == len(positions) - 1
        if is_query:
            name = plan["query_a"] if sign == 1 else plan["query_b"]
        else:
            slot = seen[sign]
            name = plan["demo_a"][slot] if sign == 1 else plan["demo_b"][slot]
            seen[sign] += 1
        out.append(
            Intervention(
                layer=LAYER,
                direction=bank[name].vector,
                strength=strength,
                positions=[position],
                per_position=True,
                label=f"{name}:{sign:+d}",
            )
        )
    return out


@torch.no_grad()
def episode_row(
    model: models.LoadedModel,
    prepared: Any,
    bank: dict[str, ConceptVector],
    plan: dict[str, Any],
    *,
    strength: float,
    seed: int,
    query_only: bool = False,
) -> dict[str, object]:
    """Score the model and every reader from one forward pass."""
    signs = prepared.episode.state_signs
    positions = prepared.state_positions
    interventions = exemplar_interventions(bank, plan, positions, signs, strength=strength)
    if query_only:
        interventions = interventions[-1:]

    ids = prepared.input_ids
    with (
        intervene(model, interventions, prompt_len=int(ids.shape[1])),
        capture(model, [LAYER]) as store,
    ):
        logits = model.forward_logits(ids)[0, -1].float()

    states = store.acts[LAYER][0][0, list(positions)].float().cpu()
    if states.shape[0] != 5:
        raise ValueError(f"expected 5 captured states, got {states.shape[0]}")

    label_ids = prepared.label_ids
    selected = logits[torch.tensor(label_ids, device=logits.device)]
    predicted = LABELS[int(selected.argmax())]
    query_sign = prepared.episode.query_sign

    row: dict[str, object] = {
        "model_predicted": predicted,
        "model_correct": predicted == prepared.episode.correct_label,
        "model_format_ok": int(logits.argmax()) in set(label_ids),
        "query_sign": query_sign,
    }
    for reader in READERS:
        row[f"reader_{reader}_correct"] = _read(states, signs, reader, seed) == query_sign
    return row


def scrambled_groups(
    a_names: list[str], b_names: list[str], seed: int
) -> tuple[list[str], list[str]]:
    """Regroup the same exemplars arbitrarily: same vectors, no category."""
    pool = list(a_names) + list(b_names)
    random.Random(seed).shuffle(pool)
    half = len(pool) // 2
    return pool[:half], pool[half:]


def _self_check() -> None:
    """The query exemplar must never appear in the demonstrations."""
    a = [f"a{i}" for i in range(8)]
    b = [f"b{i}" for i in range(8)]
    plan = assign(a, b, draw=0, same_exemplar=False)
    assert plan["query_a"] not in plan["demo_a"], "query A leaked into its demos"
    assert plan["query_b"] not in plan["demo_b"], "query B leaked into its demos"
    assert len(set(plan["demo_a"])) == 2, "demonstrations must differ from each other"

    same = assign(a, b, draw=0, same_exemplar=True)
    assert same["query_a"] == same["demo_a"][0], "anchor arm must reuse its vector"

    # Each position gets its own exemplar, and the query's is the held-out one.
    bank = {n: ConceptVector(name=n, layer=LAYER, vector=torch.randn(8)) for n in a + b}
    ivs = exemplar_interventions(bank, plan, (0, 1, 2, 3, 4), (1, -1, 1, -1, 1), strength=1.0)
    assert len(ivs) == 5, "one edit per position"
    assert ivs[-1].label.startswith(plan["query_a"]), "query carries the held-out exemplar"
    used = [i.label.split(":")[0] for i in ivs[:4]]
    assert sorted(used) == sorted(plan["demo_a"] + plan["demo_b"]), "demos use their exemplars"

    x, y = scrambled_groups(a, b, 0)
    assert sorted(x + y) == sorted(a + b) and len(x) == len(y), "scramble is a permutation"


def load_gate() -> list[str]:
    """Refuse to run without a passed capacity gate."""
    if not GATE_OUT.exists():
        raise SystemExit(
            f"{GATE_OUT} missing: run scripts/run_category_geometry.py first. "
            "Without the gate a null here cannot be told apart from injecting "
            "structure the site does not carry."
        )
    gate = json.loads(GATE_OUT.read_text())
    chosen = gate.get("frozen_choice", [])
    if len(chosen) < 2:
        raise SystemExit(
            f"gate passed only {len(chosen)} category pair(s): {chosen}. "
            "notes/23 needs two independently chosen pairs for its kill rule."
        )
    return list(chosen)


def run(args: argparse.Namespace) -> None:
    _self_check()
    out = args.out
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    for path in (out, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    chosen = load_gate()
    preflight_check(MODEL, training=False)
    model = models.load(MODEL, device=torch.device("mps"), revision=MODEL_REVISION)
    started = time.time()
    try:
        model.model.to(torch.bfloat16)
        object.__setattr__(model, "dtype", torch.bfloat16)

        pairs = chosen[:1] if args.smoke else chosen
        carriers = CONFIRM_VISIBLE_SAMPLES[:1] if args.smoke else CONFIRM_VISIBLE_SAMPLES[:2]
        draws = 1 if args.smoke else DRAWS

        rows: list[dict[str, object]] = []
        for pair_name in pairs:
            a_names, b_names = (list(x) for x in CANDIDATES[pair_name])
            bank = build_centered_bank(model, a_names + b_names)
            print(f"bank built for {pair_name}", flush=True)

            # Random directions matched to the real exemplars, one per name.
            rand_bank = {
                n: random_control(bank[n], seed=i) for i, n in enumerate(a_names + b_names)
            }

            for carrier in carriers:
                episodes = exact_episodes(carrier)
                if args.smoke:
                    episodes = episodes[:2]
                prepared_all = [prepare_episode(model, e) for e in episodes]
                carrier_sha = hashlib.sha256(carrier.encode()).hexdigest()[:16]

                for draw in range(draws):
                    # Re-scramble per draw. A single fixed permutation can land
                    # semi-coherent by luck -- the smoke drew one grouping two
                    # buildings deep -- which would weaken the control.
                    scram_a, scram_b = scrambled_groups(a_names, b_names, seed=draw)
                    plans: dict[str, tuple[dict[str, Any], dict[str, ConceptVector]]] = {
                        "same_exemplar": (
                            assign(a_names, b_names, draw=draw, same_exemplar=True),
                            bank,
                        ),
                        "heldout_semantic": (
                            assign(a_names, b_names, draw=draw, same_exemplar=False),
                            bank,
                        ),
                        "heldout_scrambled": (
                            assign(scram_a, scram_b, draw=draw, same_exemplar=False),
                            bank,
                        ),
                        "heldout_random": (
                            assign(a_names, b_names, draw=draw, same_exemplar=False),
                            rand_bank,
                        ),
                        # Only the query position is edited. With the label
                        # mapping randomized per cell and no demonstrations to
                        # read it from, this is pinned at chance unless
                        # something leaks.
                        "query_only": (
                            assign(a_names, b_names, draw=draw, same_exemplar=False),
                            bank,
                        ),
                    }
                    for arm, (plan, use_bank) in plans.items():
                        # Equalize the distance between the two demonstrated
                        # classes so arms differ in structure, not in magnitude.
                        c_a = torch.stack([use_bank[n].vector for n in plan["demo_a"]]).mean(0)
                        c_b = torch.stack([use_bank[n].vector for n in plan["demo_b"]]).mean(0)
                        strength = matched_strength(c_a, c_b)
                        # Matching pushes near-identical centroids apart by
                        # cranking strength; past a point that is no longer the
                        # same intervention, it is a much larger one.
                        if strength > MAX_MATCHED_STRENGTH:
                            raise SystemExit(
                                f"{arm}/{pair_name}/draw{draw}: matched strength "
                                f"{strength:.1f} exceeds {MAX_MATCHED_STRENGTH}. The two "
                                "demonstrated classes are nearly collinear, so equalizing "
                                "their separation would inject a far larger edit than the "
                                "other arms. Report the degenerate geometry instead."
                            )

                        for prepared in prepared_all:
                            ep = prepared.episode
                            result = episode_row(
                                model,
                                prepared,
                                use_bank,
                                plan,
                                strength=strength,
                                seed=draw,
                                query_only=(arm == "query_only"),
                            )
                            rows.append(
                                {
                                    "arm": arm,
                                    # twin_pair keys on (pair, carrier, cell_base);
                                    # the draw must be inside it or twins from
                                    # different exemplar assignments collide.
                                    "pair": f"{pair_name}|draw{draw}",
                                    "category_pair": pair_name,
                                    "draw": draw,
                                    "carrier_sha": carrier_sha,
                                    "cell_id": ep.cell_id,
                                    "cell_base": ep.cell_id.rsplit("q", 1)[0],
                                    "strength": strength,
                                    "demo_a": plan["demo_a"],
                                    "demo_b": plan["demo_b"],
                                    "query_exemplar": (
                                        plan["query_a"] if ep.query_sign == 1 else plan["query_b"]
                                    ),
                                    **result,
                                }
                            )
                    print(
                        f"  {pair_name} {carrier_sha[:8]} draw {draw}: {len(rows)} rows", flush=True
                    )

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")

        summary: dict[str, Any] = {
            "what_this_is": (
                "notes/23. Every injection position carries a different exemplar; "
                "the query exemplar appears in no demonstration. Primary metric "
                "is twin-pair accuracy, per notes/22."
            ),
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "smoke": args.smoke,
            "category_pairs": pairs,
            "draws": draws,
            "n_rows": len(rows),
            "elapsed_seconds": round(time.time() - started, 1),
            "arms": {},
        }
        for arm in ARMS:
            arm_rows = [r for r in rows if r["arm"] == arm]
            if not arm_rows:
                continue
            correct = [dict(r, correct=r["model_correct"]) for r in arm_rows]
            entry: dict[str, Any] = {
                "n": len(arm_rows),
                "row_model": sum(bool(r["model_correct"]) for r in arm_rows) / len(arm_rows),
                "twin_pair_model": twin_pair(correct),
                "format_rate": sum(bool(r["model_format_ok"]) for r in arm_rows) / len(arm_rows),
            }
            for reader in READERS:
                key = f"reader_{reader}_correct"
                entry[f"row_{reader}"] = sum(bool(r[key]) for r in arm_rows) / len(arm_rows)
                entry[f"twin_pair_{reader}"] = twin_pair(
                    [dict(r, correct=r[key]) for r in arm_rows]
                )
            entry["twin_pair_model_minus_reader"] = (
                entry["twin_pair_model"] - entry["twin_pair_centroid_euclidean"]
            )
            summary["arms"][arm] = entry

        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {out} and {summary_path}")
        for arm, e in summary["arms"].items():
            print(
                f"{arm:18s} pair:model={e['twin_pair_model']:.3f} "
                f"pair:reader={e['twin_pair_centroid_euclidean']:.3f} "
                f"row:model={e['row_model']:.3f}"
            )
    finally:
        del model
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
