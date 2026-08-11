"""Run the per-family competence screen against a real model.

This is the parent gate for Branches 04, 18 and 19: all three assume a chosen
task is actually performed. A family the model cannot execute is not a
preference option, it is noise.

    python run_family_screen.py --n-items 4 --families reverse_string,sum_numbers   # smoke
    python run_family_screen.py --out results/family_screen_qwen3-4b.json           # full
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

from binding_tasks import FAMILIES
from family_screen import screen_all, screen_report
import local_provider


def recording(complete, sink: list):
    """Wrap a provider so every prompt/reply pair is kept for inspection.

    `screen_family` returns counts; the interesting question when a family fails
    is whether the model got the work wrong or just wrapped it in prose, and only
    the raw reply answers that.
    """
    def wrapped(messages):
        t0 = time.time()
        r = complete(messages)
        sink.append({
            "prompt": messages[-1]["content"],
            "reply": r["text"],
            "seconds": round(time.time() - t0, 2),
        })
        return r
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=local_provider.DEFAULT_MODEL)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--n-items", type=int, default=16)
    ap.add_argument("--system", default=local_provider.ANSWER_PROTOCOL_SYSTEM,
                    help="uniform system prompt, recorded in the artifact; "
                         "--system '' reproduces the unprompted baseline")
    ap.add_argument("--max-new-tokens", type=int,
                    default=local_provider.MAX_NEW_TOKENS,
                    help="the ANSWER: protocol needs room for the working")
    ap.add_argument("--headroom", type=float, default=1.5,
                    help="GiB of slack above the guard's predicted peak "
                         "(default 1.5: short-context generation, not the probe's "
                         "activation capture + 512-token cache)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    raw: list = []
    complete, close = local_provider.load(args.model, system=args.system,
                                          max_new_tokens=args.max_new_tokens,
                                          headroom_gib=args.headroom)
    t0 = time.time()
    try:
        screens = screen_all(recording(complete, raw), families=families,
                             n_items=args.n_items)
    finally:
        close()

    report = screen_report(screens)
    report["model"] = args.model
    report["system_prompt"] = args.system
    report["wall_clock_s"] = round(time.time() - t0, 1)
    report["seconds_per_call"] = round((time.time() - t0) / max(len(raw), 1), 2)
    print(json.dumps(report["families"], indent=2))
    print("eligible (>=14/16):", report["eligible_acceptable"])
    print(f"{len(raw)} calls in {report['wall_clock_s']}s "
          f"({report['seconds_per_call']}s/call)")

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        out.with_suffix(".raw.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in raw))
        print("wrote", out, "and", out.with_suffix(".raw.jsonl"))


if __name__ == "__main__":
    main()
