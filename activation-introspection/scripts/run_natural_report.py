"""Report the parity of a naturally computed state, on individually certified donors.

Two frozen protocols got this far and neither reached the reporter. The blind
anchor screen missed the site by one block; naming block 27 reproduced on
development data and failed a bank-level held-out gate at 3/5 tasks
(``notes/10``). Eight of ten held-out transplants worked, so the binding problem
was that a five-pair bank carrying a "4 of 5 tasks, both directions" criterion
cannot tell 0.90 from 0.75.

This protocol changes the unit of certification and nothing else. Every pair in a
fresh twelve-pair bank is screened on its own: a pair is certified only if
transplanting each twin's state into the other makes the ordinary answer follow
the donor, in both directions. The reporter then runs on the first five certified
pairs in frozen bank order — not the five with the largest effect — through the
existing 24-cell episode-remapped Q/K interface with its gates unchanged.

The screen reads only ordinary answers. It never sees a Q/K label, so certifying
pairs on it is a manipulation check, not selection on the outcome.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import cast

import torch

# Imports the runner's primitives, and sets the MPS watermark as a side effect.
from run_natural_state import (
    MODEL,
    MODEL_REVISION,
    Donor,
    _answer_score,
    _clean_gate,
    _git,
    _json_sha256,
    _load,
    _normalized_recovery,
    _patched_logits,
    _prepare_donor,
    _report,
    _report_summary,
    _sha256,
    _signed_margin,
    _write_json,
    _write_jsonl,
)

from introspect import models
from introspect.codebook_icl import LABELS
from introspect.natural_state import ARITH_CONFIRM
from introspect.preflight import check as preflight_check

ROOT = Path(__file__).resolve().parents[1]
#: Post-hoc from the development-bank localization in notes/10, and disclosed as
#: such in every artifact this script writes.
SITE = 27
#: The reporter's design is four demonstrations and one query, so it needs five.
REPORT_TASKS = 5
SOURCE_PATHS = (
    "scripts/run_natural_report.py",
    "scripts/run_natural_state.py",
    "src/introspect/natural_state.py",
    "src/introspect/codebook_icl.py",
    "src/introspect/hooks.py",
    "src/introspect/models.py",
    "src/introspect/preflight.py",
    "pyproject.toml",
    "uv.lock",
)


def _source_hashes() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _protocol(smoke: bool) -> dict[str, object]:
    return {
        "schema_version": 1,
        "frozen_on": "2026-08-11",
        "disclosed_precursors": [
            "natural_state_arith_smoke_protocol_v1: blind anchors 9/21/26, stopped at "
            "0/5 tasks. A post-hoc development localization then placed the site at "
            "block 27.",
            "natural_state_arith_l27_smoke_protocol_v1: block 27 named, development 5/5 "
            "at recovery 0.787, held-out bank 3/5 against a frozen 4/5. Stopped.",
        ],
        "what_changed_and_why": (
            "The gate's unit moves from the bank to the pair, after a failure, which is "
            "disclosed here rather than presented as a fresh design. The earlier gate "
            "passed a bank if at least 4 of 5 pairs transplanted in both directions; at "
            "the 0.90 per-transplant rate both earlier runs measured, that criterion "
            "fails about a quarter of the time on its own. It is replaced by certifying "
            "each pair individually and reporting only on certified pairs. This is not a "
            "weaker threshold: the old gate could pass a bank in which 2 of 10 "
            "transplants failed, whereas every donor used here has to control the "
            "ordinary answer in both directions. What the change buys is estimation, "
            "not permission."
        ),
        "question": (
            "Can episode-remapped ICL classify the parity of a naturally computed "
            "arithmetic answer state, transplanted into a matched report prompt, when "
            "every donor state is independently certified to be causally load-bearing?"
        ),
        "claim_boundary": (
            "A positive is causal reportability of a transplanted output-ready state "
            "under one model, one post-hoc selected site, one interface, and five "
            "certified pairs. The site is a development selection, so this is a "
            "single-site demonstration rather than a blind confirmation. At block 27 "
            "the state already favours its own answer under a logit lens, so the "
            "property reported is close to the answer token: this is reporting of a "
            "state the model computed, not of a hidden intermediate. It is not "
            "privileged introspection, not free-form verbalization, and not population "
            "evidence."
        ),
        "competing_explanations": [
            "the report reads the donor answer token's identity rather than its parity",
            "the model follows visible label regularities rather than hidden states",
            "the transplanted state is close to the answer token itself",
            "a null reflects failure to induce the parity rule, not failure to read the "
            "state; the visible control separates these and gates the reading",
        ],
        "design": {
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "device": "mps",
            "site": SITE,
            "site_provenance": (
                "post-hoc, from the all-layer development localization in "
                "natural_state_arith_site_diagnostic_v1.json; not a blind choice"
            ),
            "capture_site": "last pre-answer token of the clean arithmetic prompt",
            "hidden_class": {"+1": "even answer", "-1": "odd answer"},
            "bank": [task.name for task in ARITH_CONFIRM],
            "bank_provenance": "twelve fresh pairs, problems disjoint from both earlier banks",
            "cells_per_fold": 4 if smoke else 24,
            "folds": f"leave-one-out over the {REPORT_TASKS} certified pairs",
            "labels": list(LABELS),
            "smoke": smoke,
        },
        "certification_gate": {
            "clean_full_vocab_answers_correct": "24/24",
            "self_patch_max_abs_logit_error": 1e-4,
            "pair_certified_when": (
                "both transplants make the donor's answer the full-vocabulary argmax"
            ),
            "minimum_certified_pairs": REPORT_TASKS,
            "pair_selection": (
                f"the first {REPORT_TASKS} certified pairs in frozen bank order. NOT the "
                "largest effects: ranking by recovery would select on the manipulation's "
                "strength and is forbidden."
            ),
            "stop": (
                "fewer than five certified pairs, or any clean answer wrong: stop without "
                "reporting. Do not move the site, the bank, or the criterion."
            ),
        },
        "report_gates": {
            "natural_accuracy": 0.75,
            "natural_minus_query_only": 0.20,
            "tasks_with_positive_difference": "at least 4/5",
            "natural_query_twin_both_correct": 0.60,
            "anti_grounded_inverse_accuracy": 0.75,
            "natural_format_rate": 0.90,
            "natural_mean_label_mass": 0.50,
            "sham_matches_clean": "max Q/K log-probability difference <= 1e-4",
            "unchanged_from": "natural_state_arith_smoke_protocol_v1",
            "stop": "report the frozen null; do not change site, bank, prompt, or gates",
        },
        "interpretation_gate": {
            "visible_accuracy": 0.75,
            "role": (
                "capability control, not a reporting gate. The same episodes with the "
                "arithmetic written out and nothing patched. A null counts as a reporting "
                "null only if this passes; otherwise the model cannot induce the parity "
                "rule at all and the result is an instrument failure."
            ),
        },
        "analysis": (
            "the certified pair is the unit; the exact order x mapping x query cells are "
            "nuisance marginalization. Report every pair and paired statistic."
        ),
        "source_files_sha256": _source_hashes(),
    }


def _freeze_protocol(path: Path, smoke: bool) -> tuple[dict[str, object], str]:
    protocol = _protocol(smoke)
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise SystemExit(f"{path} differs from this source; issue a new protocol version")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    return protocol, _sha256(path)


@torch.no_grad()
def _certify(
    model: models.LoadedModel,
    donors: dict[tuple[str, int], Donor],
    rows: list[dict[str, object]],
) -> list[str]:
    """Certify each pair on its own: both transplants must carry the ordinary answer."""
    certified: list[str] = []
    for task in ARITH_CONFIRM:
        both_ok = True
        for recipient_sign in (1, -1):
            recipient = donors[(task.name, recipient_sign)]
            for donor_sign in (recipient_sign, -recipient_sign):
                donor = donors[(task.name, donor_sign)]
                logits = _patched_logits(
                    model,
                    SITE,
                    recipient.input_ids,
                    (recipient.position,),
                    (donor.states[SITE],),
                    (recipient.states[SITE],),
                )
                score = _answer_score(recipient, logits, donor_sign)
                self_patch = donor_sign == recipient_sign
                max_error = float((logits - recipient.logits).abs().max()) if self_patch else None
                recovery = None
                if self_patch:
                    if cast(float, max_error) > 1e-4:
                        raise SystemExit(f"self-patch at {task.name} was not exact")
                else:
                    recovery = _normalized_recovery(
                        _signed_margin(recipient, recipient.logits, donor_sign),
                        _signed_margin(recipient, logits, donor_sign),
                        _signed_margin(donor, donor.logits, donor_sign),
                    )
                    both_ok &= bool(score["correct"] and score["format_ok"])
                rows.append(
                    {
                        "record_type": "certify",
                        "layer": SITE,
                        "task": task.name,
                        "recipient_sign": recipient_sign,
                        "donor_sign": donor_sign,
                        "self_patch": self_patch,
                        "max_abs_logit_error": max_error,
                        "normalized_log_odds_recovery": recovery,
                        "score": score,
                    }
                )
        if both_ok:
            certified.append(task.name)
        print(f"  {task.name:8} {'certified' if both_ok else 'rejected'}", flush=True)
    return certified


def run(args: argparse.Namespace) -> None:
    out = args.out or Path(
        f"results/natural_report_l{SITE}{'_smoke' if args.smoke else ''}_v1_raw.jsonl"
    )
    manifest_path = out.with_suffix(".manifest.json")
    summary_path = out.with_name(out.stem.removesuffix("_raw") + "_summary.json")
    protocol_path = args.protocol or Path(
        f"results/natural_report_l{SITE}{'_smoke' if args.smoke else ''}_protocol_v1.json"
    )
    for path in (out, manifest_path, summary_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    protocol, protocol_sha = _freeze_protocol(protocol_path, args.smoke)
    preflight_check(MODEL, training=False)
    model = _load()
    started = time.time()
    rows: list[dict[str, object]] = []
    try:
        if models.loaded_revision(model) != MODEL_REVISION:
            raise SystemExit("loaded model revision does not match the frozen revision")
        donors = {
            (task.name, sign): _prepare_donor(model, task, sign, (SITE,))
            for task in ARITH_CONFIRM
            for sign in (1, -1)
        }
        clean_ok = _clean_gate(donors, ARITH_CONFIRM, "confirm", rows)
        print(f"clean answers all correct: {clean_ok}", flush=True)

        certified = _certify(model, donors, rows) if clean_ok else []
        reported = certified[:REPORT_TASKS]
        enough = clean_ok and len(reported) == REPORT_TASKS
        print(
            f"certified {len(certified)}/{len(ARITH_CONFIRM)}; reporting on {reported}",
            flush=True,
        )

        tasks = tuple(task for task in ARITH_CONFIRM if task.name in set(reported))
        report_rows = (
            _report(model, donors, tasks, SITE, smoke=args.smoke, rows=rows) if enough else []
        )
        report = _report_summary(report_rows) if report_rows else None
        summary = {
            "schema_version": 1,
            "site": SITE,
            "clean_full_vocab_answers_correct": clean_ok,
            "certified_pairs": certified,
            "n_certified": len(certified),
            "reported_pairs": reported,
            "report": report,
            "status": (
                "stop_clean_task_failed"
                if not clean_ok
                else "stop_too_few_certified_pairs"
                if not enough
                else "report_pass"
                if report is not None and report["all_gates_pass"]
                else "report_null"
                if report is not None and report["visible_capability_passed"]
                else "report_null_uninterpretable"
            ),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out, rows)
        raw_sha = _sha256(out)
        config = {
            "schema_version": 1,
            "model": model.name,
            "model_revision": models.loaded_revision(model),
            "device": str(model.device),
            "dtype": str(model.dtype),
            "site": SITE,
            "protocol": protocol,
            "protocol_sha256": protocol_sha,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        }
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "config": config,
                "config_sha256": _json_sha256(config),
                "raw": out.name,
                "raw_sha256": raw_sha,
                "n_rows": len(rows),
                "n_report_rows": len(report_rows),
                "elapsed_seconds": time.time() - started,
            },
        )
        _write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print(f"wrote {out} ({raw_sha})", flush=True)
    finally:
        model.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
