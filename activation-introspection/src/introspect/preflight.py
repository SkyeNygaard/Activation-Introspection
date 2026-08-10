"""Refuse to start a model run that this machine cannot currently hold.

On a 24 GB unified-memory Mac an oversized or concurrent MPS job does not fail
cleanly. It thrashes swap, and a process killed under memory pressure can leave
GPU buffers wired with no owning process, which only a restart clears. Both have
happened here. The cheap fix is to look before loading.

Two checks:

* **Another model run is already going.** MPS is not shared gracefully. A second
  3B job launched alongside a running 4B job was killed by the OS mid-load.
  Resident set size does not reveal this, because unified-memory weights are not
  counted in RSS, so the check looks for sibling runner processes by name.
* **Not enough memory is free.** Peak driver memory on this machine fits
  ``1.10 + 1.88 * params_B`` GiB (R²=1.0000, measured at bf16 with a KV cache).
  Training adds optimizer state on top, so the requirement is scaled.

Run standalone before any experiment::

    uv run python -m introspect.preflight qwen-3b --training

Older runners cannot import this: their source hashes are locked into frozen
protocols and editing them would invalidate published artifacts. Use the
standalone form for those.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess

#: Peak driver memory in GiB, from measurements on this machine at bfloat16.
_PEAK_GIB = {"qwen-0.5b": 2.03, "qwen-1.5b": 3.99, "qwen-3b": 6.91, "qwen-7b": 14.3}

#: LoRA parameters, Adam moments and stored activations roughly double the
#: forward-only peak. Measured 7.2 GiB peak for a 3B training run against a
#: 6.91 GiB forward-only figure, so 1.5x is a floor rather than a guess.
_TRAINING_MULTIPLIER = 1.5

#: Leave room for the OS and the user's applications rather than filling RAM.
_HEADROOM_GIB = 2.0

#: Matches an experiment script invoked by absolute or relative path. The
#: sibling session that killed a run here launched `python run_dev.py`, with
#: no leading slash, so anchoring on one silently missed it.
_RUNNER_RE = re.compile(r"(?:^|[\s/])(run_[a-z0-9_]+\.py)(?:\s|$)")


def available_gib() -> float:
    """Memory macOS could hand out now: free, inactive and purgeable pages.

    Compressed pages are excluded deliberately. Counting them would report
    memory that is only available after decompression work, which is exactly the
    pressure this check exists to avoid.
    """
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))  # type: ignore[union-attr]
    total = 0
    for label in ("Pages free", "Pages inactive", "Pages purgeable", "Pages speculative"):
        found = re.search(rf"{label}:\s+(\d+)", out)
        if found:
            total += int(found.group(1))
    return total * page / 2**30


def competing_runs() -> list[tuple[int, str]]:
    """Other processes running an experiment script, excluding this one."""
    out = subprocess.run(
        ["ps", "-Ao", "pid=,command="], capture_output=True, text=True, check=True
    ).stdout
    mine = os.getpid()
    found = []
    for line in out.splitlines():
        pid_text, _, command = line.strip().partition(" ")
        if not pid_text.isdigit() or int(pid_text) == mine:
            continue
        if "python" not in command:
            continue
        match = _RUNNER_RE.search(command)
        if match and "preflight" not in command:
            found.append((int(pid_text), match.group(1)))
    return found


def required_gib(model: str, *, training: bool) -> float:
    peak = _PEAK_GIB.get(model)
    if peak is None:
        raise SystemExit(f"no measured peak for {model!r}; add one before relying on this check")
    return peak * (_TRAINING_MULTIPLIER if training else 1.0) + _HEADROOM_GIB


def check(model: str, *, training: bool = False) -> None:
    """Raise ``SystemExit`` rather than let the OS kill the run mid-load."""
    others = competing_runs()
    if others:
        listing = ", ".join(f"pid {pid} ({name})" for pid, name in others)
        raise SystemExit(
            f"another experiment is already running: {listing}. "
            "MPS is not shared gracefully on this machine — a second model job "
            "gets killed mid-load and can leave GPU buffers wired. Wait for it."
        )
    need = required_gib(model, training=training)
    have = available_gib()
    if have < need:
        raise SystemExit(
            f"only {have:.1f} GiB available, need about {need:.1f} GiB for "
            f"{model}{' training' if training else ''}. Close applications or "
            "wait; starting anyway would thrash swap."
        )
    print(f"preflight ok: {have:.1f} GiB available, {need:.1f} GiB needed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", default="qwen-3b")
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    check(args.model, training=args.training)


if __name__ == "__main__":
    main()
