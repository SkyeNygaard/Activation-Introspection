"""Recompute and verify the saved programmatic-attention benchmark summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from benchmark_programmatic_attention import _load_protocol, _sha256, _summarize


def verify(protocol_path: Path, raw_path: Path, summary_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    protocol, protocol_sha = _load_protocol(protocol_path, root)
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in raw_path.read_text().splitlines() if line
    ]
    if not rows or any(row.get("protocol_sha256") != protocol_sha for row in rows):
        raise ValueError("raw rows do not all refer to the frozen protocol")
    computed = _summarize(
        rows,
        protocol=protocol,
        protocol_sha=protocol_sha,
        raw_sha=_sha256(raw_path),
    )
    saved = json.loads(summary_path.read_text())
    if computed != saved:
        digest = hashlib.sha256(
            json.dumps(computed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        raise ValueError(f"saved summary differs from recomputation ({digest})")
    return computed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    summary = verify(args.protocol, args.raw, args.summary)
    print(
        f"verified {summary['n_raw_rows']} rows; "
        f"publishable_operator_note={summary['gates']['publishable_operator_note']}"
    )


if __name__ == "__main__":
    main()
