#!/usr/bin/env python3
"""Post-hoc secondary analysis of matched-visible latent codebook structure.

This script never imports repository analyzers.  It reads checked-in raw JSONL and
constructs a stricter four-row metric ('latent XOR quartet') that requires a model
to be correct for both hidden query signs under both hidden demonstration
conventions while the visible prompt is byte-identical.

It is explicitly a post-hoc secondary analysis, not a preregistered endpoint.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent / "extract3" / "results"
OUT = Path(__file__).resolve().parent / "latent_binding_secondary_results.json"


def load_jsonl(name: str) -> list[dict]:
    with (ROOT / name).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def cell_quartet_map() -> dict[str, str]:
    """Map every exact episode cell_id to its visible-prompt equivalence quartet."""
    rows = load_jsonl("codebook_icl_confirm_v2_raw.jsonl")
    # One concept and one carrier are enough: visible prompt equivalence depends
    # on the episode's visible labels, not the concept direction.
    base = [r for r in rows if r["concept"] == "garden" and r["carrier_id"] == 0]
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in base:
        groups[r["prompt_sha256"]].append(r)
    assert groups and {len(v) for v in groups.values()} == {4}
    mapping: dict[str, str] = {}
    for prompt_sha, rs in groups.items():
        ids = sorted(r["cell_id"] for r in rs)
        # Four rows = two query signs crossed with two hidden demo conventions.
        assert sorted(r["query_sign"] for r in rs) == [-1, -1, 1, 1]
        assert collections.Counter(r["correct_label"] for r in rs) == {"Q": 2, "K": 2}
        # Every group really is byte-identical visible text.
        assert len({r["prompt"] for r in rs}) == 1
        qid = "|".join(ids)
        for cid in ids:
            mapping[cid] = qid
    assert len(mapping) == 24
    return mapping


CELL_QUARTET = cell_quartet_map()


def rate(xs: list[bool]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def codebook(name: str) -> dict:
    rows = load_jsonl(name)
    conditions = list(rows[0]["condition_scores"])
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[(r["concept"], r["carrier_id"], r["prompt_sha256"])].append(r)
    assert {len(v) for v in groups.values()} == {4}
    out = {}
    for cond in conditions:
        vals = {k: all(bool(r["condition_scores"][cond]["correct"]) for r in rs) for k, rs in groups.items()}
        by_concept = {}
        for concept in sorted({k[0] for k in vals}):
            vv = [v for k, v in vals.items() if k[0] == concept]
            by_concept[concept] = rate(vv)
        out[cond] = {"n_quartets": len(vals), "quartet_all_correct": rate(list(vals.values())), "by_concept": by_concept}
    return out


def remap() -> dict:
    out = {}
    for seed in range(3):
        rows = load_jsonl(f"remap_training_v2_seed{seed}_raw.jsonl")
        for strength in sorted({float(r["strength"]) for r in rows}):
            for arm in sorted({r["arm"] for r in rows}):
                for cond in sorted({r["condition"] for r in rows if float(r["strength"]) == strength}):
                    rr = [r for r in rows if float(r["strength"]) == strength and r["arm"] == arm and r["condition"] == cond]
                    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
                    for r in rr:
                        groups[(r["concept"], r["carrier_sha256"], r["prompt_sha256"])].append(r)
                    groups = {k: v for k, v in groups.items() if len(v) == 4}
                    if not groups:
                        continue
                    all4 = {k: all(bool(x["correct"]) for x in v) for k, v in groups.items()}
                    ncorrect = collections.Counter(sum(bool(x["correct"]) for x in v) for v in groups.values())
                    by_concept = {}
                    for concept in sorted({k[0] for k in groups}):
                        vv = [v for k, v in all4.items() if k[0] == concept]
                        by_concept[concept] = rate(vv)
                    out[f"seed{seed}|strength{strength}|{arm}|{cond}"] = {
                        "n_quartets": len(groups),
                        "quartet_all_correct": rate(list(all4.values())),
                        "n_correct_per_quartet": dict(sorted(ncorrect.items())),
                        "by_concept": by_concept,
                    }
    # Seed-level means are descriptive; seed is the meaningful replication unit.
    aggregate = {}
    for strength in (0.15, 0.25, 0.5):
        for arm in ("base", "fixed", "remap"):
            for cond in ("target", "random"):
                keys = [f"seed{s}|strength{strength}|{arm}|{cond}" for s in range(3)]
                values = [out[k]["quartet_all_correct"] for k in keys if k in out]
                if values:
                    aggregate[f"strength{strength}|{arm}|{cond}"] = {"seed_values": values, "mean": mean(values)}
    return {"per_seed": out, "aggregate": aggregate}


def heldout() -> dict:
    rows = load_jsonl("heldout_semantic_v1_raw.jsonl")
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rr = [r for r in rows if r["arm"] == arm]
        groups: dict[tuple, list[dict]] = collections.defaultdict(list)
        for r in rr:
            groups[(r["pair"], r["carrier_sha"], CELL_QUARTET[r["cell_id"]])].append(r)
        assert {len(v) for v in groups.values()} == {4}
        entry = {"n_quartets": len(groups)}
        for key in (
            "model_correct",
            "reader_centroid_cosine_correct",
            "reader_centroid_euclidean_correct",
            "reader_shuffled_labels_correct",
        ):
            entry[key + "_quartet"] = rate([all(bool(x[key]) for x in v) for v in groups.values()])
        out[arm] = entry
    return out


def content() -> dict:
    rows = load_jsonl("matched_reader_content_v1_raw.jsonl")
    out = {}
    for arm in sorted({r["arm"] for r in rows}):
        rr = [r for r in rows if r["arm"] == arm]
        groups: dict[tuple, list[dict]] = collections.defaultdict(list)
        for r in rr:
            groups[(r["pair"], r["carrier_sha"], CELL_QUARTET[r["cell_id"]])].append(r)
        assert {len(v) for v in groups.values()} == {4}
        entry = {"n_quartets": len(groups)}
        for key in (
            "model_correct",
            "reader_centroid_cosine_correct",
            "reader_centroid_euclidean_correct",
            "reader_shuffled_labels_correct",
        ):
            entry[key + "_quartet"] = rate([all(bool(x[key]) for x in v) for v in groups.values()])
        out[arm] = entry
    return out


def main() -> None:
    result = {
        "status": "POST_HOC_SECONDARY_ANALYSIS_NOT_PREREGISTERED",
        "metric_definition": (
            "A latent-XOR quartet is four rows with byte-identical visible prompt: two hidden demonstration "
            "configurations encode opposite latent sign-to-label conventions, crossed with the two hidden query signs. "
            "Quartet success requires all four outputs correct. A deterministic visible-only strategy, query-only hidden "
            "strategy, or demonstration-only hidden strategy has structural quartet success 0."
        ),
        "codebook_test": codebook("codebook_icl_test_raw.jsonl"),
        "codebook_confirm": codebook("codebook_icl_confirm_v2_raw.jsonl"),
        "remap_training_v2": remap(),
        "heldout_semantic": heldout(),
        "matched_reader_content": content(),
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
