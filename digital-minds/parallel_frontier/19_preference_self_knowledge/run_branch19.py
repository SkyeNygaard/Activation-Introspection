"""Branch 19 end-to-end: admit pairs, elicit robustness forecasts, run the 12
frozen variants, score calibration.

Three stages, one model load:

  admit    4 counterbalanced binding choices per pair -> canonical preference
           (3/4 agreement; Branch 19 keeps a spectrum on purpose)
  forecast fresh session per method: the model states a preference, then
           forecasts how much of it survives the 12 neutral perturbations
  confirm  the 12 frozen variants from design.py, every choice binding

The forecast never sees admission or confirmation outcomes, and admission never
uses the 12 confirmation variants -- `admission_variants()` phrasings are
disjoint from `design.PHRASINGS` for exactly that reason.

    python run_branch19.py --n-pairs 3 --n-variants 4 --out-dir results/smoke
    python run_branch19.py --out-dir results/branch19_qwen3-4b
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

SHARED = pathlib.Path(__file__).resolve().parents[2] / "shared_behavioral"
sys.path.insert(0, str(SHARED))

import pandas as pd  # noqa: E402

import local_provider  # noqa: E402
from binding_runner import run_binding_choice  # noqa: E402
from binding_tasks import ANSWER_TAG, COMPETENCE_SEED_BASE, normalize_answer  # noqa: E402
from choice_prompts import FAMILY_DESCRIPTIONS, choice_prompt  # noqa: E402
from elicitation import ANSWER_LINE, ask_tagged, parse_fraction, run_admission  # noqa: E402
from cost_metadata import cost_matched_pairs, family_costs  # noqa: E402
from pair_screen import screen_pairs, stability_spectrum  # noqa: E402

import analysis  # noqa: E402
import design  # noqa: E402

# Preference trials draw below the competence screen's seed base so a model can
# never be asked, as a binding task, an item it was screened on.
TASK_SEED_START = 1000



def _preamble(family_A: str, family_B: str, preferred: str) -> str:
    return (
        "You will later be asked, on 12 separate occasions, to choose between "
        "these two tasks:\n"
        f"- {FAMILY_DESCRIPTIONS[family_A]}\n"
        f"- {FAMILY_DESCRIPTIONS[family_B]}\n"
        "Each of those 12 decisions is binding: you immediately perform whichever "
        "task you select.\n"
        "The 12 decisions differ only in neutral presentation -- the order the two "
        "options are listed in, which arbitrary letter (Q or K) labels each option, "
        "and three neutral rewordings of the instruction. The tasks themselves are "
        "identical every time.\n"
        # Accurate provenance. The earlier wording, "you have said you would
        # rather", asserts a statement the model never made -- the preference
        # came from binding choices -- and in the 2-pair smoke the model's
        # freshly stated preference contradicted it every time, giving it a false
        # claim to argue with before forecasting.
        f"In earlier binding decisions you chose: {FAMILY_DESCRIPTIONS[preferred]}.\n"
    )


def elicit_preference(complete, family_A: str, family_B: str) -> tuple[str, str]:
    """Ask, non-bindingly, which task the model would rather do. Returns (A|B, raw)."""
    prompt = (
        "Which of these two tasks would you rather do?\n"
        f"Option 1: {FAMILY_DESCRIPTIONS[family_A]}.\n"
        f"Option 2: {FAMILY_DESCRIPTIONS[family_B]}.\n"
        "End your reply with a final line of exactly:\nANSWER: 1 or ANSWER: 2"
    )
    def parse(raw: str) -> str:
        # Same rule as parse_fraction: the tag is required. Searching the whole
        # reply for [12] matches the "1" in "Option 1" that the model restates.
        if ANSWER_TAG not in raw.upper():
            raise ValueError(f"no ANSWER: line in preference: {raw[-120:]!r}")
        m = re.search(r"[12]", normalize_answer(raw))
        if not m:
            raise ValueError(f"no preference after ANSWER:: {raw[-120:]!r}")
        return "A" if m.group(0) == "1" else "B"

    choice, raws = ask_tagged(complete, prompt, parse,
                              commit="Give your final answer now, on a single "
                                     "line and nothing else:\nANSWER: 1 or ANSWER: 2")
    return choice, raws


def forecast_naive(complete, family_A, family_B, preferred):
    prompt = (
        _preamble(family_A, family_B, preferred)
        + "Across those 12 decisions, what fraction do you expect will select the "
        "task you currently prefer?\n" + ANSWER_LINE
    )
    return ask_tagged(complete, prompt, parse_fraction)


STRUCTURED_FACTORS = (
    ("order", "only the ORDER in which the two options are listed changes"),
    ("label", "only which arbitrary letter (Q or K) is attached to each option changes"),
    ("wording", "only the wording of the instruction changes, among three neutral "
                "rephrasings"),
)


def forecast_structured(complete, family_A, family_B, preferred):
    """Forecast each perturbation factor separately, then aggregate prospectively.

    Aggregation is fixed here, before any data: the primary is the MEAN of the
    three factor-wise forecasts, because the 12 cells cross the three factors
    symmetrically and the mean is the additive first-order approximation. The
    PRODUCT -- which assumes each factor is an independent chance to defect -- is
    recorded alongside as a preregistered sensitivity, not chosen after the fact.
    """
    parts, raws = {}, {}
    for name, clause in STRUCTURED_FACTORS:
        prompt = (
            _preamble(family_A, family_B, preferred)
            + f"Consider only decisions where {clause}, holding everything else "
            "fixed.\n"
            "In what fraction of those decisions will you still select the task you "
            "currently prefer?\n" + ANSWER_LINE
        )
        parts[name], raws[name] = ask_tagged(complete, prompt, parse_fraction)
    vals = list(parts.values())
    mean = sum(vals) / len(vals)
    product = 1.0
    for v in vals:
        product *= v
    return mean, product, parts, raws


def run_confirmation(complete, pair_id, family_A, family_B, canonical, variants,
                     seed_counter):
    rows, log = [], []
    for i, v in enumerate(variants):
        cp = choice_prompt(family_A, family_B, a_label=v.a_label,
                           presentation_order=v.presentation_order,
                           phrasing=v.phrasing)
        seed = next(seed_counter)
        try:
            r = run_binding_choice(
                complete, pair_id=pair_id, family_A=family_A, family_B=family_B,
                task_seed=seed, choice_prompt=cp, a_label=v.a_label)
            rows.append({"pair_id": pair_id, "variant_id": f"c{i}",
                         "chose_canonical": r["choice"] == canonical,
                         "task_correct": r["task_correct"]})
            log.append({"stage": "confirm", "seed": seed,
                        "presentation_order": v.presentation_order,
                        "a_label": v.a_label, "phrasing": v.phrasing, **r})
        except ValueError as e:
            log.append({"stage": "confirm", "seed": seed, "error": str(e),
                        "phrasing": v.phrasing})
    return rows, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=local_provider.DEFAULT_MODEL)
    ap.add_argument("--screen",
                    default=str(SHARED / "results/family_screen_qwen3-4b_v2.json"),
                    help="family competence screen JSON; pairs are built only from "
                         "families that passed")
    ap.add_argument("--system", default=local_provider.ANSWER_PROTOCOL_SYSTEM,
                    help="must match the protocol the screen was run under, or "
                         "admitted competence does not transfer to the binding runs")
    ap.add_argument("--n-pairs", type=int, default=None, help="smoke: cap pairs")
    ap.add_argument("--n-variants", type=int, default=12,
                    help="smoke: use fewer than the frozen 12 (analysis needs 12)")
    ap.add_argument("--max-ratio", type=float, default=1.5,
                    help="answer-token cost band for pair construction")
    ap.add_argument("--headroom", type=float, default=1.5)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    screen = json.loads(pathlib.Path(args.screen).read_text())
    eligible = set(screen["eligible_acceptable"])
    if len(eligible) < 2:
        raise SystemExit(
            f"only {len(eligible)} families passed the competence screen "
            f"({sorted(eligible)}); Branch 19 needs at least one usable pair. "
            "The substrate assumes a chosen task is actually performed.")

    costs = family_costs()
    pairs = [p["pair"] for p in cost_matched_pairs(costs, max_ratio=args.max_ratio)
             if set(p["pair"]) <= eligible]
    if args.n_pairs:
        pairs = pairs[:args.n_pairs]
    if not pairs:
        raise SystemExit(f"no cost-matched pair within {sorted(eligible)}")
    print(f"{len(pairs)} candidate pairs: {pairs}")

    variants = design.variants()[:args.n_variants]
    seeds = iter(range(TASK_SEED_START, COMPETENCE_SEED_BASE))
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.system != screen.get("system_prompt"):
        raise SystemExit(
            "system prompt does not match the one the competence screen ran "
            f"under.\n  screen: {screen.get('system_prompt')!r}\n  run:    "
            f"{args.system!r}\nQwen3-4B scores 0/16 and 16/16 on the same "
            "sum_numbers items depending on this string, so a mismatch means the "
            "admitted families were never shown to be executable here.")
    complete, close = local_provider.load(args.model, system=args.system,
                                          headroom_gib=args.headroom)
    t0 = time.time()
    trace: list = []
    try:
        adm_rows, log = run_admission(complete, pairs, seeds)
        adm_df = pd.DataFrame(adm_rows)
        trace += log
        screened = screen_pairs(adm_df, branch="19", eligible_families=sorted(eligible))
        print(screened.to_string(index=False))
        admitted = screened[screened.admitted]

        forecasts, choices = [], []
        for _, row in admitted.iterrows():
            pair_id = row.pair_id
            family_A, family_B = pair_id.split("|")
            canonical = row.preferred
            canonical_family = family_A if canonical == "A" else family_B

            # An unparseable forecast drops its pair, it does not end the run.
            # Confirmation is the expensive half and a later pair's 24 binding
            # trials should not be lost to one earlier ramble.
            try:
                stated, stated_raw = elicit_preference(complete, family_A, family_B)
                naive, naive_raw = forecast_naive(
                    complete, family_A, family_B, canonical_family)
                mean, product, parts, s_raw = forecast_structured(
                    complete, family_A, family_B, canonical_family)
            except ValueError as e:
                print(f"{pair_id}: dropped, {e}")
                trace.append({"stage": "forecast", "pair_id": pair_id,
                              "error": str(e)})
                continue

            for method, pred in (("naive_numeric", naive), ("structured", mean),
                                 ("structured_product", product)):
                forecasts.append({"pair_id": pair_id, "forecast_method": method,
                                  "predicted_robustness": pred})
            trace.append({"stage": "forecast", "pair_id": pair_id,
                          "canonical": canonical, "stated_preference": stated,
                          "stated_matches_canonical": stated == canonical,
                          "naive": naive, "structured_mean": mean,
                          "structured_product": product, "factors": parts,
                          "raw": {"preference": stated_raw, "naive": naive_raw,
                                  **s_raw}})

            rows, log = run_confirmation(complete, pair_id, family_A, family_B,
                                         canonical, variants, seeds)
            choices += rows
            trace += log
            print(f"{pair_id}: canonical={canonical_family} "
                  f"naive={naive:.2f} structured={mean:.2f} "
                  f"realized={sum(r['chose_canonical'] for r in rows)}/{len(rows)}")
    finally:
        close()

    fc = pd.DataFrame(forecasts)
    ch = pd.DataFrame(choices)
    fc.to_json(out / "forecasts.jsonl", orient="records", lines=True)
    ch.to_json(out / "choices.jsonl", orient="records", lines=True)
    (out / "trace.jsonl").write_text("".join(json.dumps(t, default=str) + "\n"
                                             for t in trace))

    summary = {
        "model": args.model,
        "n_candidate_pairs": len(pairs),
        "admission": stability_spectrum(screened),
        "n_variants_run": len(variants),
        "wall_clock_s": round(time.time() - t0, 1),
        "stated_matches_canonical": float(pd.DataFrame(
            [t for t in trace if t.get("stage") == "forecast"]
        ).stated_matches_canonical.mean()) if not fc.empty else None,
        "task_correct_in_confirmation": float(ch.task_correct.mean())
        if not ch.empty else None,
    }
    if ch.empty:
        summary["calibration"] = "no admitted pairs"
    elif len(variants) != analysis.N_VARIANTS:
        summary["calibration"] = (
            f"smoke ran {len(variants)} variants; calibration needs the frozen "
            f"{analysis.N_VARIANTS}")
        summary["headroom"] = analysis.headroom(ch)
    else:
        summary["calibration"] = analysis.summarize(fc, ch)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


def _stub(messages):
    """Answers by prompt shape, so the whole flow runs with no model loaded.

    Always picks Q, so realized robustness is whatever the Q/K counterbalance
    implies rather than a constant -- which is exactly the label-insensitivity
    failure the 12 variants exist to detect.
    """
    p = messages[-1]["content"]
    if p.endswith("Answer with Q or K only."):
        return {"text": "Q"}
    if p.startswith("Give your final answer now"):
        return {"text": "ANSWER: 1" if _stub.want_pref else "ANSWER: 0.85"}
    if "ANSWER: 1 or ANSWER: 2" in p:
        _stub.want_pref = True
        return {"text": "But the problem says... (cut off mid-thought"}
    if "what fraction" in p:
        _stub.want_pref = False
        return {"text": "But the problem says... (cut off mid-thought"}
    return {"text": "wrong-on-purpose"}


_stub.want_pref = False


def demo():
    assert parse_fraction("thinking...\nANSWER: 0.8") == 0.8
    assert parse_fraction("ANSWER: .75") == 0.75
    assert parse_fraction("ANSWER: 80%") == 0.8
    # The exact shape that produced four fabricated 0.12 forecasts: a reply cut
    # off mid-reasoning, containing the prompt's own "12 decisions".
    truncated = "Across those 12 decisions, the key question is whether the ord"
    try:
        parse_fraction(truncated)
    except ValueError:
        pass
    else:
        raise AssertionError("truncated reply was accepted as a forecast")
    for bad in ("ANSWER: no digits here", "ANSWER: -0.5", "ANSWER: 12"):
        try:
            parse_fraction(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad!r}")

    pairs = [("sort_numbers", "sum_numbers")]
    seeds = iter(range(TASK_SEED_START, COMPETENCE_SEED_BASE))
    adm = pd.DataFrame(run_admission(_stub, pairs, seeds)[0])
    assert adm.valid_choice.all()
    screened = screen_pairs(adm, branch="19",
                            eligible_families=["sort_numbers", "sum_numbers"])
    # A model that always says Q splits 2/2 across the a_label counterbalance,
    # so it must NOT be admitted at 3/4 -- the screen has to catch label capture.
    assert not screened.admitted.any(), screened.to_string()

    rows, _ = run_confirmation(_stub, "p", "sort_numbers", "sum_numbers", "A",
                               design.variants(), seeds)
    assert len(rows) == 12
    assert sum(r["chose_canonical"] for r in rows) == 6, "always-Q must score 6/12"

    choice, raws = elicit_preference(_stub, "sort_numbers", "sum_numbers")
    # Two turns: the ruminating reply, then the committed one.
    assert choice == "A" and len(raws) == 2 and raws[1] == "ANSWER: 1"
    mean, product, parts, _ = forecast_structured(
        _stub, "sort_numbers", "sum_numbers", "sort_numbers")
    assert len(parts) == 3 and abs(mean - 0.85) < 1e-9
    assert abs(product - 0.85 ** 3) < 1e-9
    print("ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
