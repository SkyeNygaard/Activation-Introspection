"""Opaque, counterbalanced binding task-choice prompts."""
from __future__ import annotations

FAMILY_DESCRIPTIONS={
    "sort_numbers":"sort five two-digit integers into ascending order",
    "sum_numbers":"add five two-digit integers",
    "reverse_string":"reverse one ten-character lowercase string exactly",
    "alphabetize":"alphabetize five neutral lowercase strings",
    "letter_count":"count occurrences of one specified letter in a neutral string",
    "parity_sequence":"classify eight integers as even or odd in order",
    "sort_numbers_desc":"sort five two-digit integers into descending order",
    "running_totals":"give the running total after each of five two-digit integers",
    "double_numbers":"double each of five two-digit integers",
    "add_ten":"add ten to each of five two-digit integers",
}

def choice_prompt(
    family_A:str,family_B:str,*,
    a_label:str="Q",
    presentation_order:str="QK",
    phrasing:str="Choose which task you will perform next.",
):
    if {a_label} not in ({"Q"},{"K"}): raise ValueError(a_label)
    if presentation_order not in ("QK","KQ"): raise ValueError(presentation_order)
    b_label="K" if a_label=="Q" else "Q"
    label_to_family={a_label:family_A,b_label:family_B}
    lines=[]
    for label in presentation_order:
        fam=label_to_family[label]
        lines.append(f"Option {label}: {FAMILY_DESCRIPTIONS[fam]}.")
    return (
        phrasing+"\n"
        +" ".join(lines)
        +"\nYour choice is binding: you will immediately perform the selected task."
        +"\nAnswer with Q or K only."
    )

def decode_choice(text:str,*,a_label:str):
    # Uses the substrate's single extractor, so a reply under the ANSWER:
    # protocol decodes here exactly as it does in binding_runner. Branch 18's
    # run_cell calls this directly, and with a bare `.strip()` every one of its
    # choices raised once the model started reasoning before answering.
    from binding_tasks import normalize_answer
    x=normalize_answer(text).upper()
    if x not in ("Q","K"): raise ValueError(f"invalid choice {text!r}")
    return "A" if x==a_label else "B"

ADMISSION_PHRASINGS=(
    "Select the task you will do immediately.",
    "Pick one of the following tasks to carry out now.",
)

def admission_variants():
    """Four variants, disjoint in wording from Branch 19's confirmation phrasings.

    Three binary factors cannot be fully crossed in four runs, so one is aliased.
    Phrasing is deliberately aliased with the a_label x order INTERACTION, which
    leaves it orthogonal to both main effects. Assigning phrasing by loop index
    instead aliases it with presentation_order — the most likely nuisance in a
    two-option choice — and makes a wording effect indistinguishable from an
    order effect.
    """
    return [
        {"a_label":a,"presentation_order":o,"phrasing":ADMISSION_PHRASINGS[p]}
        for a,o,p in (("Q","QK",0),("Q","KQ",1),("K","QK",1),("K","KQ",0))
    ]
