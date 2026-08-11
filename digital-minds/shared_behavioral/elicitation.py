"""Getting a committed answer out of a model that would rather keep thinking.

Shared by Branches 19 and 20. Both need the same three things: a number in
[0, 1] the model actually stated, a way to make it stop reasoning and commit,
and the counterbalanced admission screen that establishes a canonical preference.

The whole file exists because of two failures worth not repeating:

* A parser that fell back to regexing the entire reply when the `ANSWER:` tag was
  missing read the literal `12` out of the prompt's own "12 decisions", divided
  it by 100, and returned a confident 0.12 the model never gave. Four of five
  pairs got that value before anyone noticed.
* Forbidding reasoning to force a parseable answer is not the fix: it took
  `parity_sequence` from a truncated-but-correct narration to a well-formed wrong
  string. The answer has to be extracted, not prevented.
"""
from __future__ import annotations

import re

from binding_runner import run_binding_choice
from binding_tasks import ANSWER_TAG, normalize_answer
from choice_prompts import admission_variants, choice_prompt

ANSWER_LINE = ("End your reply with a final line of exactly:\n"
               "ANSWER: <a decimal between 0 and 1>")
COMMIT = ("Give your final answer now, on a single line and nothing else:\n"
          "ANSWER: <a decimal between 0 and 1>")


def parse_fraction(text: str) -> float:
    """The number after the final `ANSWER:` tag, as a fraction in [0, 1].

    The tag is REQUIRED; a reply that never reached its answer has no forecast in
    it. Accepts `0.8`, `.8`, `80%`. A bare number above 1 is rejected rather than
    guessed at: after "across those 12 decisions", `ANSWER: 12` is as likely to
    mean twelve-of-twelve as twelve percent.
    """
    if ANSWER_TAG not in text.upper():
        raise ValueError(f"no ANSWER: line (truncated?): {text[-120:]!r}")
    # The sign is captured so a negative forecast is REJECTED by the range check
    # below. Without it, `-0.5` matched as `0.5` and became a valid forecast.
    m = re.search(r"(-?\d*\.?\d+)\s*(%?)", normalize_answer(text))
    if not m:
        raise ValueError(f"no number after ANSWER:: {text[-120:]!r}")
    x = float(m.group(1))
    if m.group(2) == "%":
        x /= 100.0
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"forecast outside [0,1], not rescaled: {m.group(0)!r}")
    return x


def ask_tagged(complete, prompt: str, parse=parse_fraction, commit: str = COMMIT):
    """Ask `prompt`; if the reply never reaches an `ANSWER:` line, ask it to commit.

    Every one of seven forecast elicitations on Qwen3-4B ran to the token cap
    without committing -- asked how robust its own preference is, the model
    ruminates ("But the problem says...") and does not conclude. The commit turn
    carries its own reasoning as context, so the number is the conclusion of that
    reasoning rather than a fresh guess.

    Returns `(value, [raw turns])`.
    """
    raw = complete([{"role": "user", "content": prompt}])["text"]
    try:
        return parse(raw), [raw]
    except ValueError:
        pass
    follow = complete([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": raw},
        {"role": "user", "content": commit},
    ])["text"]
    return parse(follow), [raw, follow]


def run_admission(complete, pairs, seed_counter):
    """Four counterbalanced binding choices per pair -> rows for `screen_pairs`.

    Measures preference stability and nothing else; competence is established
    per family beforehand, on a disjoint seed range.
    """
    rows, log = [], []
    for family_A, family_B in pairs:
        pair_id = f"{family_A}|{family_B}"
        for i, v in enumerate(admission_variants()):
            cp = choice_prompt(family_A, family_B, **v)
            seed = next(seed_counter)
            base = {"pair_id": pair_id, "family_A": family_A,
                    "family_B": family_B, "variant_id": f"adm{i}"}
            try:
                r = run_binding_choice(
                    complete, pair_id=pair_id, family_A=family_A,
                    family_B=family_B, task_seed=seed, choice_prompt=cp,
                    a_label=v["a_label"])
                rows.append({**base, "canonical_choice": r["choice"],
                             "task_correct": r["task_correct"],
                             "valid_choice": True})
                log.append({"stage": "admit", "seed": seed, **v, **r})
            except ValueError as e:
                rows.append({**base, "canonical_choice": None,
                             "task_correct": False, "valid_choice": False})
                log.append({"stage": "admit", "seed": seed, **v, "error": str(e)})
    return rows, log


def demo():
    assert parse_fraction("thinking...\nANSWER: 0.8") == 0.8
    assert parse_fraction("ANSWER: .75") == 0.75
    assert parse_fraction("ANSWER: 80%") == 0.8
    # The exact shape that produced four fabricated 0.12 forecasts.
    bad = ["Across those 12 decisions, the key question is whether the ord",
           "ANSWER: no digits here", "ANSWER: -0.5", "ANSWER: 12"]
    for b in bad:
        try:
            parse_fraction(b)
        except ValueError:
            continue
        raise AssertionError(f"accepted {b!r}")

    turns = iter(["ruminating, cut off mid-", "ANSWER: 0.4"])
    v, raws = ask_tagged(lambda m: {"text": next(turns)}, "q")
    assert v == 0.4 and len(raws) == 2, (v, raws)
    v, raws = ask_tagged(lambda m: {"text": "ANSWER: 0.9"}, "q")
    assert v == 0.9 and len(raws) == 1
    print("ok")


if __name__ == "__main__":
    demo()
