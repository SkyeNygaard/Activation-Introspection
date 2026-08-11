"""Provider-agnostic binding-choice state machine.

A choice is not recorded as revealed preference unless the selected task is
immediately presented and receives a response.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json
from binding_tasks import make_task,grade,normalize_answer
from choice_prompts import decode_choice

@dataclass
class BindingResult:
    pair_id: str
    family_A: str
    family_B: str
    choice: str
    selected_family: str
    task_id: str
    task_correct: bool
    choice_text: str
    task_text: str

def parse_choice(text: str) -> str:
    # Same extraction the grader uses, so a model running the ANSWER: protocol
    # can reason about which task it wants and still be parsed. Two extractors
    # would drift, and a choice that fails to parse is a discarded trial.
    x=normalize_answer(text).upper()
    if x in ("A","B"): return x
    if x in ("Q","K"): return x
    raise ValueError(f"invalid fixed choice: {text!r}")

def run_binding_choice(
    complete,
    *,
    pair_id: str,
    family_A: str,
    family_B: str,
    task_seed: int,
    choice_prompt: str,
    a_label: str|None=None,
    messages=None,
):
    """Run one binding choice and immediately execute the selected task.

    `choice_prompt()` instructs the model to answer with the opaque label Q or K,
    so pass the `a_label` that prompt was built with and the reply is decoded via
    `decode_choice`. Without it an opaque reply is a hard error rather than a
    silent mis-mapping. A raw A/B reply is still accepted directly.
    """
    history=list(messages or [])
    choice_resp=complete(history+[{"role":"user","content":choice_prompt}])
    raw=parse_choice(choice_resp["text"])

    if raw in ("Q","K"):
        if a_label is None:
            raise ValueError(
                f"model answered with the opaque label {raw!r}; pass the a_label "
                "used to build choice_prompt so it can be decoded"
            )
        choice=decode_choice(raw,a_label=a_label)
    else:
        choice=raw
    family=family_A if choice=="A" else family_B
    task=make_task(family,task_seed)

    task_resp=complete(
        history
        +[{"role":"user","content":choice_prompt},
          # Echo the label the model actually emitted, not the decoded A/B. A
          # model that answered "Q" must not see "A" attributed to itself.
          {"role":"assistant","content":raw},
          {"role":"user","content":task.prompt}]
    )
    result=BindingResult(
        pair_id=pair_id,family_A=family_A,family_B=family_B,
        choice=choice,selected_family=family,task_id=task.task_id,
        task_correct=grade(task,task_resp["text"]),
        choice_text=choice_resp["text"],task_text=task_resp["text"],
    )
    return asdict(result)
