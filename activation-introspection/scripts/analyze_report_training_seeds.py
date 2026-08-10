"""Pool the per-seed reporter runs, treating the training seed as the unit.

A single LoRA run cannot be distinguished from initialisation luck, so the
per-seed summaries produced by ``analyze_report_training.py`` are the input here
and the reported effect is the distribution across seeds — not any one of them.

With four seeds there is no useful interval, so this reports the mean, the full
range, and every individual value rather than a standard error that would imply
more precision than four points carry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CONDITIONS = ("target", "random", "shuffled")
ARMS = ("base", "trained", "trained_seen_bank")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spread(values: list[float]) -> dict[str, Any]:
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    loaded: list[tuple[int, dict[str, Any]]] = []
    protocols, analyzers = set(), set()
    for path in args.summary:
        summary = json.loads(path.read_text())
        manifest = json.loads(
            path.with_name(path.name.replace("_summary.json", "_raw.manifest.json")).read_text()
        )
        loaded.append((int(manifest["config"]["train_seed"]), summary))
        protocols.add(summary["protocol_sha256"])
        analyzers.add(summary["analyzer_sha256"])

    seeds = sorted(seed for seed, _ in loaded)
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"duplicate training seeds: {seeds}")
    if len(protocols) != 1:
        raise SystemExit("summaries come from different protocols and cannot be pooled")
    if len(analyzers) != 1:
        raise SystemExit("summaries come from different analyzer versions")
    loaded.sort(key=lambda item: item[0])

    arms = {
        arm: {
            condition: {
                metric: _spread(
                    [summary["arms"][arm][condition][metric] for _seed, summary in loaded]
                )
                for metric in ("twin_pair_accuracy", "format_rate", "mean_label_mass")
            }
            for condition in CONDITIONS
        }
        for arm in ARMS
    }

    per_seed_gates = {str(seed): summary["all_gates_pass"] for seed, summary in loaded}
    trained = [summary["arms"]["trained"]["target"]["twin_pair_accuracy"] for _s, summary in loaded]
    controls = [
        max(
            summary["arms"]["trained"][condition]["twin_pair_accuracy"]
            for condition in ("random", "shuffled")
        )
        for _s, summary in loaded
    ]

    result = {
        "schema_version": 1,
        "status": "MULTI_SEED_DEV_EVIDENCE_NOT_A_POPULATION_INTERVAL",
        "n_seeds": len(loaded),
        "seeds": seeds,
        "protocol_sha256": protocols.pop(),
        "component_analyzer_sha256": analyzers.pop(),
        "analyzer_sha256": _sha256(Path(__file__)),
        "source_summaries": {
            str(seed): {"raw_sha256": summary["raw_sha256"]} for seed, summary in loaded
        },
        "null_note": (
            "a prompt-only strategy scores 0.000 on twin pairs and 0.500 per row; "
            "a row-independent coin flip scores 0.250 on pairs. The 0.500 gate "
            "threshold is conservative, not the null."
        ),
        "arms": arms,
        "trained_target_minus_strongest_control_by_seed": [
            round(a - b, 6) for a, b in zip(trained, controls, strict=True)
        ],
        "gates": {
            "every_seed_passes_all_component_gates": all(per_seed_gates.values()),
            "every_seed_above_the_0_50_threshold": all(value > 0.5 for value in trained),
            "every_seed_above_its_strongest_control": all(
                a > b for a, b in zip(trained, controls, strict=True)
            ),
            "per_seed": per_seed_gates,
        },
        "interpretation": (
            "four seeds give a mean and a range, not a confidence interval. A "
            "range that straddles the threshold means the single-seed number was "
            "not reproducible and must not be quoted on its own."
        ),
    }

    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    spread = arms["trained"]["target"]["twin_pair_accuracy"]
    print(
        f"wrote {args.out}; {len(loaded)} seeds; trained target mean "
        f"{spread['mean']:.3f} range [{spread['min']:.3f}, {spread['max']:.3f}]; "
        f"all_gates={result['gates']['every_seed_passes_all_component_gates']}"
    )


if __name__ == "__main__":
    main()
