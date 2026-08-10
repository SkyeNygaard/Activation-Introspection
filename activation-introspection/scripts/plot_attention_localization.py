"""Plot the verified DEV attention-output interchange summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "27c8af5f4917dc3c72214caece71ac996d9c26ec914dd390f86f954a73e41427"
RAW_SHA256 = "530f4f550e514cb64787d3b8206742533f2e70eab1839ed018e9c24bd84d5c1c"
SUMMARY_SHA256 = "f232927533aacb0df638544c08420bcd5cbb0804feccfaa0e1ee0de956d3dcbb"
ANALYZER_SHA256 = "025a0addecc93b110d56e8123a8e75ac91681baaf83f87ca9cbbd105b2fa6d2c"
THRESHOLD = 0.20
ROLES = {
    "demo_labels": ("demo labels", "#4477AA", "o", "-"),
    "query_marker": ("query marker", "#EE7733", "s", "-"),
    "final_answer": ("final answer", "#228833", "^", "-"),
    "all_positions": ("all positions (diagnostic)", "#777777", "D", "--"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_verified(
    summary_path: Path,
    raw_path: Path,
    manifest_path: Path,
    protocol_path: Path,
    analyzer_path: Path,
) -> dict[str, Any]:
    if _sha256(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("refusing protocol with an unexpected SHA-256")
    if _sha256(raw_path) != RAW_SHA256:
        raise ValueError("refusing raw artifact with an unexpected SHA-256")
    if _sha256(summary_path) != SUMMARY_SHA256:
        raise ValueError("refusing summary with an unexpected SHA-256")
    if _sha256(analyzer_path) != ANALYZER_SHA256:
        raise ValueError("refusing summary whose analyzer source has drifted")

    protocol = cast(dict[str, Any], json.loads(protocol_path.read_text()))
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    summary = cast(dict[str, Any], json.loads(summary_path.read_text()))
    config = cast(dict[str, Any], manifest["config"])
    if (
        config.get("protocol") != protocol
        or config.get("protocol_sha256") != PROTOCOL_SHA256
        or manifest.get("raw") != raw_path.name
        or manifest.get("raw_sha256") != RAW_SHA256
        or _json_sha256(config) != manifest.get("config_sha256")
        or summary.get("protocol_sha256") != PROTOCOL_SHA256
        or summary.get("raw_sha256") != RAW_SHA256
        or summary.get("config_sha256") != manifest.get("config_sha256")
        or summary.get("analyzer_sha256") != ANALYZER_SHA256
    ):
        raise ValueError("summary, raw artifact, manifest, and protocol are not bound")
    if summary.get("design_validation") != {
        "cells": 12,
        "layers": 26,
        "roles": 4,
        "patches": 1248,
    }:
        raise ValueError("summary does not contain the verified 12 x 26 x 4 DEV design")
    return summary


def _plot(summary: dict[str, Any], out: Path) -> None:
    layers = list(range(10, 36))
    selected = cast(dict[str, list[int]], summary["selection"])
    layer_role = cast(dict[str, dict[str, dict[str, object]]], summary["layer_role"])

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for role, (label, color, marker, linestyle) in ROLES.items():
        values = [
            float(
                cast(
                    float,
                    layer_role[role][str(layer)]["aggregate_margin_removal_fraction"],
                )
            )
            for layer in layers
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"non-finite margin-removal fraction for {role}")
        ax.plot(
            layers,
            values,
            label=label,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.7,
            linestyle=linestyle,
            alpha=0.95 if role != "all_positions" else 0.78,
        )
        chosen = selected.get(role, [])
        if chosen:
            chosen_values = [
                float(
                    cast(
                        float,
                        layer_role[role][str(layer)]["aggregate_margin_removal_fraction"],
                    )
                )
                for layer in chosen
            ]
            ax.scatter(
                chosen,
                chosen_values,
                s=115,
                marker="*",
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

    ax.axhline(THRESHOLD, color="#222222", linewidth=1.1, linestyle=":")
    ax.text(
        10.15,
        THRESHOLD + 0.025,
        "20% DEV threshold",
        fontsize=8.5,
        color="#222222",
        ha="left",
        va="bottom",
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#BBBBBB",
            markeredgecolor="black",
            markersize=10,
            label="selected for next screen",
        )
    )
    labels.append("selected for next screen")
    ax.legend(handles, labels, loc="upper left", fontsize=8.3, frameon=False, ncol=2)
    ax.set_xlabel("downstream layer")
    ax.set_ylabel("aggregate margin-removal fraction")
    ax.set_xticks(list(range(10, 36, 2)))
    ax.set_xlim(9.6, 35.4)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("DEV attention-output interchange screen", fontsize=12, weight="semibold", pad=12)
    fig.text(
        0.5,
        0.015,
        "one DEV concept x one carrier; selection only, not confirmation",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "results/attention_localization_dev_v2_summary.json"
    )
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "results/attention_localization_dev_v2_raw.jsonl"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/attention_localization_dev_v2_raw.manifest.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "results/attention_localization_dev_protocol_v2.json",
    )
    parser.add_argument(
        "--analyzer", type=Path, default=ROOT / "scripts/analyze_attention_localization.py"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "figures/attention_localization_dev.png")
    args = parser.parse_args()
    summary = _load_verified(args.summary, args.raw, args.manifest, args.protocol, args.analyzer)
    _plot(summary, args.out)
    print(f"wrote {args.out} (sha256={_sha256(args.out)})")


if __name__ == "__main__":
    main()
