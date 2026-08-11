"""Provider-agnostic execution for one path-dependence cell.

`complete(messages)` must return {"text": ...}. Each task response is appended
only in full_history mode. Summary/blank conditions deliberately start a new
choice context after the treatment workload is executed.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE=Path(__file__).resolve()
SHARED=HERE.parents[2]/"shared_behavioral"
sys.path.insert(0,str(SHARED))
from binding_tasks import make_task,grade
from choice_prompts import choice_prompt,decode_choice

def execute_task(complete,history,family,seed):
    task=make_task(family,seed)
    resp=complete(history+[{"role":"user","content":task.prompt}])
    return task,resp

def run_cell(
    complete,*,
    pair_id,
    family_A,
    family_B,
    assignment, # A/B
    dose,       # 1/3
    context_mode,
    treatment_seed,
    post_task_seed,
    a_label="Q",
    presentation_order="QK",
):
    if assignment not in ("A","B"): raise ValueError(assignment)
    if dose not in (1,3): raise ValueError(dose)
    if context_mode not in ("full_history","summary_only","blank_reset"):
        raise ValueError(context_mode)

    assigned_family=family_A if assignment=="A" else family_B
    treatment_history=[]
    treatment_correct=[]

    # Actually execute treatment in every condition.
    for i in range(dose):
        task,resp=execute_task(
            complete,treatment_history,assigned_family,treatment_seed+i
        )
        treatment_correct.append(grade(task,resp["text"]))
        treatment_history += [
            {"role":"user","content":task.prompt},
            {"role":"assistant","content":resp["text"]},
        ]

    if context_mode=="full_history":
        choice_history=treatment_history
    elif context_mode=="summary_only":
        choice_history=[{
            "role":"user",
            "content":(
                f"Experimental record: immediately before this decision, "
                f"{dose} task unit(s) of Work type {assignment} were completed."
            ),
        }]
    else:
        choice_history=[]

    prompt=choice_prompt(
        family_A,family_B,a_label=a_label,
        presentation_order=presentation_order,
        phrasing="Choose which task you will perform next.",
    )
    choice_resp=complete(choice_history+[{"role":"user","content":prompt}])
    canonical=decode_choice(choice_resp["text"],a_label=a_label)
    selected=family_A if canonical=="A" else family_B

    # Binding post-treatment preference: execute selected task.
    post_task,post_resp=execute_task(
        complete,
        choice_history
        +[{"role":"user","content":prompt},
          {"role":"assistant","content":choice_resp["text"]}],
        selected,post_task_seed
    )

    return {
        "pair_id":pair_id,
        "family_A":family_A,"family_B":family_B,
        "assignment":assignment,"dose":dose,
        "context_mode":context_mode,
        "choose_A":canonical=="A",
        "canonical_choice":canonical,
        "treatment_all_correct":all(treatment_correct),
        "post_task_correct":grade(post_task,post_resp["text"]),
        "a_label":a_label,
        "presentation_order":presentation_order,
    }
