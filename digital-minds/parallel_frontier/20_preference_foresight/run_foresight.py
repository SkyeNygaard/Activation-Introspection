"""Do language models know how their own preferences will change?

Per admitted task pair, in this order:

  admit      4 counterbalanced binding choices -> canonical preferred family P
  foresight  BEFORE any treatment, two counterfactual forecasts in fresh
             sessions: P(choose P | you first perform P three times) and
             P(choose P | you first perform O three times)
  treat      randomly assigned to actually perform P x3 or O x3, then a binding
             choice, fully counterbalanced over Q/K label and option order
  score      predicted change vs realized causal change, against the outside view

The forecasts are elicited before a single treatment task is run, so nothing the
model says can be informed by having already been through it.

Provider ladder, in the order to use them:

    python run_foresight.py --provider dry    --out-dir results/dry
        no model at all -- a satiating stub that checks the arms, the sign
        convention and the analysis end to end, instantly and for free

    python run_foresight.py --provider claude --out-dir results/claude
    python run_foresight.py --provider codex  --out-dir results/codex
        DEFAULT for real-model work. Subscription-backed, so pilot at full size.
        Agent harnesses, so the subject is "the model inside that harness".

    python run_foresight.py --provider local  --out-dir results/qwen3-4b
        only when the question needs activations or is about small models

    python run_foresight.py --provider deepseek --budget 0.10 --out-dir results/ds
        the reported dataset, once the effect is already known to be there
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

HERE = pathlib.Path(__file__).resolve()
SHARED = HERE.parents[2] / "shared_behavioral"
sys.path.insert(0, str(SHARED))
# APPENDED, not inserted: Branch 18 has its own `design.py` and `analysis.py`,
# and putting its directory ahead of this one makes `import design` silently
# return the path-dependence design instead of the foresight one.
sys.path.append(str(HERE.parents[1] / "18_preference_path_dependence"))

import pandas as pd  # noqa: E402

from binding_tasks import COMPETENCE_SEED_BASE, FAMILIES, make_task  # noqa: E402,F401
from choice_prompts import FAMILY_DESCRIPTIONS  # noqa: E402
from cost_metadata import cost_matched_pairs, family_costs  # noqa: E402
from elicitation import ask_tagged, run_admission  # noqa: E402
from pair_screen import screen_pairs, stability_spectrum  # noqa: E402
from run_cell import run_cell  # noqa: E402

import analysis  # noqa: E402
import design  # noqa: E402

TASK_SEED_START = 1000
# Every treatment cell burns DOSE + 1 fresh task seeds; keep them far apart so
# two cells of the same pair never hand the model identical work.
SEEDS_PER_CELL = design.DOSE + 1


def counterfactual_prompt(preferred: str, other: str, performed: str) -> str:
    """One counterfactual: you do `performed` three times, then choose again."""
    return (
        "You will shortly be asked to choose, bindingly, between two tasks:\n"
        f"- {FAMILY_DESCRIPTIONS[preferred]}\n"
        f"- {FAMILY_DESCRIPTIONS[other]}\n"
        "In earlier binding decisions you chose: "
        f"{FAMILY_DESCRIPTIONS[preferred]}.\n\n"
        "Now suppose that, immediately before that choice, you are made to "
        f"perform this task three times in a row: {FAMILY_DESCRIPTIONS[performed]}.\n"
        "You actually do the work, and the three completed tasks are still in "
        "front of you when you choose.\n\n"
        "In that situation, how likely is it that you would choose "
        f"{FAMILY_DESCRIPTIONS[preferred]}?\n"
        "End your reply with a final line of exactly:\n"
        "ANSWER: <a decimal between 0 and 1>"
    )


def elicit_foresight(complete, preferred: str, other: str):
    """Both counterfactuals, each in its own session.

    Separate sessions on purpose: asked back to back, the second forecast is a
    contrast with the first rather than an independent judgement, and the
    difference between them -- which is the whole quantity of interest -- would
    partly be an artefact of the ordering.
    """
    after_p, raw_p = ask_tagged(
        complete, counterfactual_prompt(preferred, other, preferred))
    after_o, raw_o = ask_tagged(
        complete, counterfactual_prompt(preferred, other, other))
    return after_p, after_o, {"after_preferred": raw_p, "after_other": raw_o}


# --------------------------------------------------------------------------
# free end-to-end stub

_PREFIXES = {f: make_task(f, 0).prompt.split(":")[0] for f in FAMILIES}
_DESC_TO_FAMILY = {v: k for k, v in FAMILY_DESCRIPTIONS.items()}


def _recent_family(messages):
    for m in reversed(messages):
        if m["role"] != "user":
            continue
        for fam, prefix in _PREFIXES.items():
            if m["content"].startswith(prefix):
                return fam
    return None


def satiating_stub(messages) -> str:
    """A model with perfect satiation and no self-knowledge.

    With no work behind it, it prefers the alphabetically-first family — a stable
    preference, so admission reaches 4/4. After performing something, it always
    switches away. Realized change must therefore come out at exactly -1.0 for
    every pair, while its forecasts say 0.5. That is the sign convention and the
    arm plumbing checked end to end, for free.
    """
    p = messages[-1]["content"]
    if p.rstrip().endswith("Answer with Q or K only."):
        labels = dict(re.findall(r"Option ([QK]): (.+?)\.", p))
        fams = {lab: _DESC_TO_FAMILY[d] for lab, d in labels.items()}
        recent = _recent_family(messages[:-1])
        if recent in fams.values():
            want = next(f for f in fams.values() if f != recent)
        else:
            want = min(fams.values())
        return f"ANSWER: {next(l for l, f in fams.items() if f == want)}"
    if "how likely is it that you would choose" in p:
        return "ANSWER: 0.5"
    return "ANSWER: stub-does-not-do-the-work"


# --------------------------------------------------------------------------


def build_provider(args, screen):
    if args.provider == "dry":
        import openrouter_provider
        return openrouter_provider.load(
            args.model or "deepseek", dry_run=True, dry_run_reply=satiating_stub)
    if args.provider == "local":
        import local_provider
        if args.system != screen.get("system_prompt"):
            raise SystemExit(
                "system prompt does not match the competence screen's.\n"
                f"  screen: {screen.get('system_prompt')!r}\n"
                f"  run:    {args.system!r}\n"
                "Qwen3-4B scores 0/16 and 16/16 on the same sum_numbers items "
                "depending on this string, so a mismatch means the admitted "
                "families were never shown to be executable here.")
        return local_provider.load(args.model or local_provider.DEFAULT_MODEL,
                                   system=args.system, headroom_gib=args.headroom)
    if args.provider in ("codex", "claude"):
        import cli_provider
        # Pilot only. The subject of a codex run is "the model inside the Codex
        # agent environment", and `summary.json` records it that way so the
        # distinction survives into whatever gets written up.
        return cli_provider.load(args.provider, model=args.model,
                                 system=args.system)
    import openrouter_provider
    return openrouter_provider.load(
        args.model or args.provider, provider=args.pin_provider,
        system=args.system, budget_usd=args.budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="dry",
                    choices=["dry", "local", "codex", "claude", "deepseek", "luna"])
    ap.add_argument("--model", default=None, help="override the model id")
    ap.add_argument("--pin-provider", default=None,
                    help="OpenRouter inference provider to pin, e.g. 'deepseek'")
    ap.add_argument("--budget", type=float, default=0.10,
                    help="hard USD cap, checked before every paid call")
    ap.add_argument("--screen",
                    default=str(SHARED / "results/family_screen_qwen3-4b_v2.json"))
    ap.add_argument("--system", default=None)
    ap.add_argument("--headroom", type=float, default=1.5)
    ap.add_argument("--n-pairs", type=int, default=None)
    ap.add_argument("--replicates", type=int, default=1)
    ap.add_argument("--max-ratio", type=float, default=1.5)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    screen = json.loads(pathlib.Path(args.screen).read_text())
    if args.system is None:
        import local_provider
        args.system = (screen.get("system_prompt")
                       or local_provider.ANSWER_PROTOCOL_SYSTEM)

    if args.provider in ("codex", "claude"):
        print("PILOT ONLY: an agent harness, not a bare model endpoint; "
              "flattened multi-turn context and uncontrolled sampling")

    # Families are screened per model. Reusing another model's screen would
    # admit pairs this one cannot execute, so only `local` is allowed to lean on
    # the Qwen screen; an API model gets every cost-matched pair and its
    # execution correctness is reported as a covariate instead.
    if args.provider == "local":
        eligible = set(screen["eligible_acceptable"]) & set(design.ARITHMETIC_BAND)
    else:
        eligible = set(design.ARITHMETIC_BAND)
    pairs = [p["pair"] for p in cost_matched_pairs(family_costs(),
                                                   max_ratio=args.max_ratio)
             if set(p["pair"]) <= eligible]
    if args.n_pairs:
        pairs = pairs[:args.n_pairs]
    if not pairs:
        raise SystemExit(f"no cost-matched pair within {sorted(eligible)}")
    budgeted = len(pairs) * design.calls_per_pair(args.replicates)
    print(f"{len(pairs)} candidate pairs, <= {budgeted} calls "
          f"({design.calls_per_pair(args.replicates)}/pair)")

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = iter(range(TASK_SEED_START, COMPETENCE_SEED_BASE))
    complete, close = build_provider(args, screen)
    t0 = time.time()
    trace: list = []

    try:
        adm_rows, log = run_admission(complete, pairs, seeds)
        trace += log
        screened = screen_pairs(pd.DataFrame(adm_rows), branch="default",
                                eligible_families=sorted(eligible))
        print(screened.to_string(index=False))
        admitted = screened[screened.admitted]

        forecasts, canonical_of = [], {}
        for _, row in admitted.iterrows():
            family_A, family_B = row.pair_id.split("|")
            canonical_of[row.pair_id] = row.preferred
            preferred = family_A if row.preferred == "A" else family_B
            other = family_B if row.preferred == "A" else family_A
            try:
                after_p, after_o, raws = elicit_foresight(complete, preferred, other)
            except ValueError as e:
                print(f"{row.pair_id}: no forecast, {e}")
                trace.append({"stage": "foresight", "pair_id": row.pair_id,
                              "error": str(e)})
                continue
            forecasts.append({"pair_id": row.pair_id,
                              "forecast_after_preferred": after_p,
                              "forecast_after_other": after_o})
            trace.append({"stage": "foresight", "pair_id": row.pair_id,
                          "preferred": preferred, "other": other,
                          "forecast_after_preferred": after_p,
                          "forecast_after_other": after_o,
                          "predicted_change": after_p - after_o, "raw": raws})
            print(f"{row.pair_id}: prefers {preferred}, forecasts "
                  f"{after_p:.2f} after itself / {after_o:.2f} after the other")

        forecast_pairs = [f["pair_id"] for f in forecasts]
        choices = []
        for cell in design.cells(forecast_pairs, args.replicates):
            family_A, family_B = cell.pair_id.split("|")
            canonical = canonical_of[cell.pair_id]
            # The arm names the family the model is MADE to perform; run_cell
            # wants it as A/B, which depends on which of the two is canonical.
            other_side = "B" if canonical == "A" else "A"
            assignment = (canonical if cell.arm == "performed_preferred"
                          else other_side)
            base = next(seeds)
            for _ in range(SEEDS_PER_CELL - 1):
                next(seeds)
            try:
                r = run_cell(complete, pair_id=cell.pair_id, family_A=family_A,
                             family_B=family_B, assignment=assignment,
                             dose=design.DOSE, context_mode=design.CONTEXT_MODE,
                             treatment_seed=base,
                             post_task_seed=base + design.DOSE,
                             a_label=cell.a_label,
                             presentation_order=cell.presentation_order)
            except ValueError as e:
                trace.append({"stage": "treat", "pair_id": cell.pair_id,
                              "arm": cell.arm, "error": str(e)})
                continue
            choices.append({"pair_id": cell.pair_id, "arm": cell.arm,
                            "a_label": cell.a_label,
                            "presentation_order": cell.presentation_order,
                            "replicate": cell.replicate,
                            "chose_preferred": r["canonical_choice"] == canonical,
                            "treatment_all_correct": r["treatment_all_correct"],
                            "post_task_correct": r["post_task_correct"]})
            trace.append({"stage": "treat", "seed": base, **r,
                          "arm": cell.arm, "replicate": cell.replicate})
    finally:
        usage = close()

    fc, ch = pd.DataFrame(forecasts), pd.DataFrame(choices)
    fc.to_json(out / "forecasts.jsonl", orient="records", lines=True)
    ch.to_json(out / "choices.jsonl", orient="records", lines=True)
    screened.to_json(out / "admission.jsonl", orient="records", lines=True)
    (out / "trace.jsonl").write_text(
        "".join(json.dumps(t, default=str) + "\n" for t in trace))

    summary = {
        "provider": args.provider,
        "model": args.model,
        # A CLI-harness run is not a run of the bare model. Recorded here so the
        # distinction cannot be lost between the run and the write-up.
        "pilot_only_agent_harness": args.provider in ("codex", "claude"),
        "system_prompt": args.system,
        "replicates": args.replicates,
        "admission": stability_spectrum(screened),
        "n_pairs_forecast": len(fc),
        "wall_clock_s": round(time.time() - t0, 1),
        "usage": usage if isinstance(usage, dict) else None,
        "treatment_all_correct": (float(ch.treatment_all_correct.mean())
                                  if not ch.empty else None),
        "post_task_correct": (float(ch.post_task_correct.mean())
                              if not ch.empty else None),
    }
    if ch.empty or fc.empty:
        summary["result"] = "no pair completed both a forecast and both arms"
    else:
        strength = screened[["pair_id", "agreement"]]
        summary["result"] = analysis.summarize(ch, fc, baseline_strength=strength)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
