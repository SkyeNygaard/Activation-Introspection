"""Does the model's in-context self-report ride on the shared "something moved" axis?

Pre-run note: ``notes/39-what-does-the-model-actually-use.md``, written before this ran.

The content task from notes/14-15 is built so that "was something pushed in, and
which way" cannot produce the right answer: both options are pushed in positively
and only their content differs. The model scores 0.899 on it anyway. This asks what
that score is made of, by removing the one direction that carries "an injection
happened" from the readout and scoring the identical task again.

The direction comes from notes/38 and is fitted on eight concepts that appear
nowhere in this task, so the ablation cannot be quietly deleting the answer.

Nothing is reimplemented here: the episode scoring comes from
``run_matched_reader_content``, the task constants from ``run_content_vs_disturbance``,
and the direction fit from ``run_displacement_direction``. The ablation is applied
by wrapping their scoring in an outer ``intervene`` block -- forward hooks compose,
so the layer-9 injection and the layer-9 capture inside ``episode_row`` still fire
exactly as they did in the original run, and nothing hash-bound is edited.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from run_content_vs_disturbance import (
    LAYER,
    MODEL,
    MODEL_REVISION,
    PAIRS,
    matched_strength,
)
from run_matched_reader_content import READERS, episode_row

from introspect import models
from introspect.codebook_icl import (
    CONFIRM_CONCEPTS,
    CONFIRM_VISIBLE_SAMPLES,
    exact_episodes,
    prepare_episode,
)
from introspect.concepts import ConceptVector, build_bank
from introspect.hooks import Intervention, intervene
from introspect.preflight import check as preflight_check
from introspect.report_training import CENTERING_CONCEPTS

#: Declared before the run. notes/38 measured 1.000 at 0.5B and at 4B; this model
#: has never been checked. Below this the kill rule in notes/39 fires.
GATE_AUROC = 0.90

CONDITIONS = ("none", "displacement", "random")


def _load(name: str) -> Any:
    """Import a sibling script by path; they are scripts, not an installed package."""
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(f"_{name.removesuffix('.py')}", path)
    mod = importlib.util.module_from_spec(cast(Any, spec))
    cast(Any, spec).loader.exec_module(mod)
    return mod


DISP = _load("run_displacement_direction.py")


def _twin_pairs(rows: list[dict[str, object]], key: str) -> tuple[int, int]:
    """Score at the protocol's unit: both members of a byte-identical pair correct.

    A pair is the two episodes sharing a demonstration order and label mapping and
    differing only in the sign of the query state -- ``o0m0q+1`` with ``o0m0q-1``.
    A strategy that ignores the hidden state answers both the same way and scores 0.
    """
    groups: dict[tuple[str, str, str], list[bool]] = defaultdict(list)
    for r in rows:
        cell = str(r["cell_id"])
        base = cell.rsplit("q", 1)[0]
        groups[(str(r["pair"]), str(r["carrier_sha"]), base)].append(bool(r[key]))
    complete = [v for v in groups.values() if len(v) == 2]
    return sum(1 for v in complete if all(v)), len(complete)


def gate(model: object, bank: dict[str, ConceptVector], read_layer: int) -> dict[str, object]:
    """Fit the shared direction on development rows, score it on held-out ones."""
    dev_clean, dev_injected, _ = DISP.collect(
        model,
        bank,
        DISP.DEV_CONCEPTS,
        DISP.DEV_CARRIERS,
        inject_layer=LAYER,
        read_layer=read_layer,
        strength=1.0,
    )
    out_clean, out_injected, _ = DISP.collect(
        model,
        bank,
        DISP.HELDOUT_CONCEPTS,
        DISP.HELDOUT_CARRIERS,
        inject_layer=LAYER,
        read_layer=read_layer,
        strength=1.0,
    )
    direction = dev_injected.mean(0) - dev_clean.mean(0)
    return {
        "direction": direction,
        "separation": DISP.separation(direction, out_clean, out_injected),
        "share": DISP.displacement_share(out_clean, out_injected),
    }


def score_condition(
    model: models.LoadedModel,
    bank: dict[str, ConceptVector],
    ablation: Intervention | None,
    *,
    pairs: tuple[tuple[str, str], ...],
    carriers: tuple[str, ...],
    max_episodes: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    extra = [ablation] if ablation is not None else []
    for carrier in carriers:
        episodes = exact_episodes(carrier)
        if max_episodes is not None:
            episodes = episodes[:max_episodes]
        prepared_all = [prepare_episode(model, e) for e in episodes]
        carrier_sha = __import__("hashlib").sha256(carrier.encode()).hexdigest()[:16]
        for name_a, name_b in pairs:
            a, b = bank[name_a], bank[name_b]
            strength = matched_strength(a.vector, b.vector)
            for seed, prepared in enumerate(prepared_all):
                # The outer block is the manipulation. Everything inside it --
                # the layer-9 injection and the layer-9 capture -- is untouched.
                with intervene(model, extra):
                    row = episode_row(model, prepared, a, b, strength=strength, seed=seed)
                rows.append(
                    {
                        "pair": f"{name_a}|{name_b}",
                        "carrier_sha": carrier_sha,
                        "cell_id": prepared.episode.cell_id,
                        "strength": strength,
                        **row,
                    }
                )
        print(f"  {carrier_sha}: {len(rows)} rows", flush=True)
    return rows


def summarise(rows: list[dict[str, object]]) -> dict[str, object]:
    n = len(rows)
    model_pairs, n_pairs = _twin_pairs(rows, "model_correct")
    reader_pairs, _ = _twin_pairs(rows, "reader_centroid_euclidean_correct")
    return {
        "n": n,
        "model_accuracy": sum(bool(r["model_correct"]) for r in rows) / n,
        "model_twin_pair": model_pairs / max(n_pairs, 1),
        "n_twin_pairs": n_pairs,
        "reader_accuracy": {
            reader: sum(bool(r[f"reader_{reader}_correct"]) for r in rows) / n
            for reader in READERS
        },
        "reader_twin_pair": reader_pairs / max(n_pairs, 1),
        "format_rate": sum(bool(r["model_format_ok"]) for r in rows) / n,
        # The failure mode this repository keeps hitting: a confident collapse to
        # one label whatever the state. 0.5 is balanced; 0 or 1 is constant.
        "predicted_Q_rate": sum(1 for r in rows if r["model_predicted"] == "Q") / n,
        "by_pair": {
            pair: sum(bool(r["model_correct"]) for r in rows if r["pair"] == pair)
            / len([r for r in rows if r["pair"] == pair])
            for pair in sorted({str(r["pair"]) for r in rows})
        },
    }


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
        read_layer = model.n_layers - 1

        # The task bank, built exactly as the original content run builds it: the
        # centering offset comes from a separate concept set, not from these eight.
        centering = build_bank(model, LAYER, list(CENTERING_CONCEPTS), center=False)
        center = torch.stack([cv.vector for cv in centering.values()]).mean(dim=0)
        raw = build_bank(model, LAYER, list(CONFIRM_CONCEPTS), center=False)
        bank = {
            name: ConceptVector(name=name, layer=LAYER, vector=cv.vector - center)
            for name, cv in raw.items()
        }
        print("task bank built", flush=True)

        # The direction is fitted on notes/38's concepts, which are disjoint from
        # the task's, so this bank is built separately and never mixed with it.
        disp_bank = build_bank(
            model, LAYER, DISP.DEV_CONCEPTS + DISP.HELDOUT_CONCEPTS, center=True
        )
        g = gate(model, disp_bank, read_layer)
        auroc = float(g["separation"]["auroc"])  # type: ignore[index]
        share = g["share"]
        print(f"gate: auroc {auroc:.3f}  share {share}", flush=True)
        if auroc < GATE_AUROC:
            summary_path.write_text(
                json.dumps(
                    {
                        "gate_failed": True,
                        "auroc": auroc,
                        "share": share,
                        "threshold": GATE_AUROC,
                        "reading": "no coherent shared direction at this model; notes/39 kill rule fired",
                    },
                    indent=2,
                )
            )
            raise SystemExit(f"gate failed: auroc {auroc:.3f} < {GATE_AUROC}; stopping as declared")

        direction = cast(torch.Tensor, g["direction"])
        generator = torch.Generator(device="cpu").manual_seed(0)
        random_dir = torch.randn(direction.shape, generator=generator).to(
            direction.device, direction.dtype
        )

        pairs = PAIRS[:1] if args.smoke else PAIRS
        carriers = CONFIRM_VISIBLE_SAMPLES[:1] if args.smoke else CONFIRM_VISIBLE_SAMPLES
        max_episodes = 4 if args.smoke else None

        rows: list[dict[str, object]] = []
        summaries: dict[str, object] = {}
        for condition in CONDITIONS:
            print(f"condition: {condition}", flush=True)
            ablation = None
            if condition != "none":
                ablation = Intervention(
                    layer=read_layer,
                    direction=direction if condition == "displacement" else random_dir,
                    mode="ablate",
                    positions="last",
                    label=f"ablate_{condition}",
                )
            got = score_condition(
                model,
                bank,
                ablation,
                pairs=pairs,
                carriers=carriers,
                max_episodes=max_episodes,
            )
            for r in got:
                r["condition"] = condition
            rows.extend(got)
            summaries[condition] = summarise(got)

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        summary_path.write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "model_revision": MODEL_REVISION,
                    "inject_layer": LAYER,
                    "read_layer": read_layer,
                    "smoke": args.smoke,
                    "gate": {"auroc": auroc, "share": share, "threshold": GATE_AUROC},
                    "conditions": summaries,
                    "elapsed_s": round(time.time() - started, 1),
                    "reading": (
                        "displacement well below none, random close to none -> the model's "
                        "answer leans on the shared 'something moved' axis. displacement close "
                        "to none -> it reads concept-specific structure. both below none -> "
                        "generic damage, report the bound and stop."
                    ),
                },
                indent=2,
            )
        )
        print(json.dumps(summaries, indent=2), flush=True)
    finally:
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/readout_ablation_icl_v1_raw.jsonl"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
