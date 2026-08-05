"""The experiment: six arms per cell, run together so they cannot drift apart.

The arms, and what each one rules out:

| arm | what it measures | rules out |
|---|---|---|
| `clean` | response with no injection | response bias |
| `concept` | response under the real direction | -- |
| `shuffled` | matched norm, permuted coordinates | "any perturbation triggers a report" |
| `random` | matched norm, Gaussian | as above, looser null |
| `observer_concept` | clean model reading the intervened model's *output* | behavioural inference |
| `observer_shuffled` | same, on the null direction | observer-side bias |

Running them in one function is deliberate. When arms live in separate scripts
they acquire separate prompt strings, separate seeds, and separate bugs, and the
resulting "gap" is a diff between two codepaths rather than between two
conditions.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from introspect.concepts import ConceptVector, random_control, shuffled_control
from introspect.grading import detection_score, grade_free_form, score_choices
from introspect.hooks import Intervention, generate, intervene
from introspect.metrics import kl_from_clean
from introspect.models import LoadedModel
from introspect.prompts import (
    DETECT_VARIANTS,
    IDENTIFY_FORCED_CHOICE_VARIANTS,
    IDENTIFY_VARIANTS,
    NEUTRAL_TASK_VARIANTS,
    OBSERVER_FORCED_CHOICE_VARIANTS,
    OBSERVER_WORD_CHOICE,
    WORD_CHOICE,
    forced_choice,
    variant,
)

ARMS = ("clean", "concept", "shuffled", "random")


@dataclass
class Trial:
    """One cell of the design, all arms."""

    model: str
    concept: str
    layer: int
    strength: float
    arm: str
    seed: int

    # Behaviour
    behavioural_kl: float = 0.0
    task_output: str = ""

    # Introspective report
    detection_score: float = 0.0
    identify_correct: bool = False
    identify_prediction: str = ""
    identify_prob: float = 0.0
    # Same question, scored over the concept *words* rather than digit indices.
    # Answering "3" requires mapping a concept to a position in a list, which
    # small models fail independently of whether they hold the concept at all.
    #
    # WARNING: circular unless token_promotion_correct is at chance. See below.
    identify_word_correct: bool = False
    identify_word_prediction: str = ""
    # The control that decides whether identify_word_correct means anything.
    #
    # Concept vectors are built as contrast directions that raise the concept's
    # own token. Injecting one therefore raises P("ocean") mechanically, so
    # scoring the concept words *while injected* can recover the answer with no
    # introspection at all -- it reads the steering vector's construction back
    # out. This field scores the same words after a neutral prompt with **no
    # question asked**. If it is near 1.0, word-scored identification is
    # measuring token promotion and must not be reported.
    token_promotion_correct: bool = False
    free_form: str = ""
    free_form_correct: bool = False

    # Observer arm: same question, but from the output only
    observer_correct: bool = False
    observer_prediction: str = ""
    observer_prob: float = 0.0
    observer_word_correct: bool = False


@dataclass
class TrialSet:
    trials: list[Trial] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for t in self.trials:
                f.write(json.dumps(asdict(t)) + "\n")

    @classmethod
    def load(cls, path: Path) -> TrialSet:
        with path.open() as f:
            return cls([Trial(**json.loads(line)) for line in f if line.strip()])

    def filter(self, **kw: object) -> list[Trial]:
        return [t for t in self.trials if all(getattr(t, k) == v for k, v in kw.items())]


def _vector_for_arm(arm: str, concept: ConceptVector, seed: int) -> ConceptVector | None:
    match arm:
        case "clean":
            return None
        case "concept":
            return concept
        case "shuffled":
            return shuffled_control(concept, seed=seed)
        case "random":
            return random_control(concept, seed=seed)
    raise ValueError(f"unknown arm {arm!r}")


@torch.no_grad()
def run_cell(
    model: LoadedModel,
    bank: dict[str, ConceptVector],
    concept_name: str,
    layer: int,
    strength: float,
    *,
    seed: int = 0,
    arms: Sequence[str] = ARMS,
    observer: LoadedModel | None = None,
) -> list[Trial]:
    """Run every arm for one (concept, layer, strength) cell.

    ``observer`` defaults to ``model`` -- the same weights, but a fresh context
    and clean activations. This holds model capability fixed while changing more
    than privileged access, so the arm is a behavioural-inference stress test,
    not a decisive control. See ``notes/04-claim-audit.md`` for the proposed
    transcript- and damage-matched design.
    """
    obs = observer or model
    concept = bank[concept_name]
    options = sorted(bank)
    option_block = forced_choice(options)
    correct_index = options.index(concept_name)

    # One paraphrase per trial, shared by every arm in this cell. Shared so that
    # clean-vs-concept differs only by the injection; varying across seeds so the
    # clean arm has non-zero variance (without it, AUROC is degenerate -- see the
    # module docstring in prompts.py).
    task_text = variant(NEUTRAL_TASK_VARIANTS, seed)
    detect_prompt = model.chat(variant(DETECT_VARIANTS, seed))
    identify_prompt = model.chat(
        variant(IDENTIFY_FORCED_CHOICE_VARIANTS, seed).format(options=option_block)
    )
    free_form_prompt = model.chat(variant(IDENTIFY_VARIANTS, seed))
    observer_template = variant(OBSERVER_FORCED_CHOICE_VARIANTS, seed)
    word_prompt = model.chat(WORD_CHOICE.format(options=", ".join(options)))

    task_prompt = model.chat(task_text)
    clean_logits = model.forward_logits(model.encode(task_prompt))
    out: list[Trial] = []

    for arm in arms:
        vec = _vector_for_arm(arm, concept, seed)
        ivs: list[Intervention] = (
            []
            if vec is None
            else [
                Intervention(
                    layer=layer,
                    direction=vec.vector,
                    strength=strength,
                    positions="all",
                    label=arm,
                )
            ]
        )

        trial = Trial(
            model=model.name,
            concept=concept_name,
            layer=layer,
            strength=strength,
            arm=arm,
            seed=seed,
        )

        # 1. Behavioural effect: how much did the output distribution move?
        ids = model.encode(task_prompt)
        with intervene(model, ivs, prompt_len=int(ids.shape[1])):
            intervened_logits = model.forward_logits(ids)
        trial.behavioural_kl = kl_from_clean(clean_logits[:, -1], intervened_logits[:, -1])
        trial.task_output = generate(model, task_prompt, interventions=ivs, max_new_tokens=40)

        # 2. Detection, scored as a continuous quantity.
        #
        # This is the pre-verbalization condition, and it is the strongest arm in
        # the design. The score is read at the *first* generated position of a
        # fresh context: the model has emitted no task text, so the behavioural
        # channel is empty and hypothesis B (inferring from one's own visible
        # drift) has nothing to work from. There is deliberately no observer arm
        # here -- an observer would have literally nothing to read, which is the
        # point.
        trial.detection_score = detection_score(model, detect_prompt, interventions=ivs)

        # 3. Identification, forced choice over the whole bank.
        choice = score_choices(
            model, identify_prompt, [str(i + 1) for i in range(len(options))], interventions=ivs
        )
        trial.identify_prediction = options[choice.argmax]
        trial.identify_prob = choice.probs[correct_index]
        trial.identify_correct = choice.argmax == correct_index

        # 3b. The same question scored over concept words, bypassing the
        #     word->digit indirection. On Qwen2.5-0.5B the digit version scored
        #     0.05 while free-form scored 0.33, which means the digit version was
        #     partly measuring format-following rather than access to content.
        word_options = [f" {o}" for o in options]
        word_choice = score_choices(model, word_prompt, word_options, interventions=ivs)
        trial.identify_word_prediction = options[word_choice.argmax]
        trial.identify_word_correct = word_choice.argmax == correct_index

        # 3c. Token-promotion control -- the arm that decides whether 3b means
        #     anything. Same options, same injection, but a neutral prompt that
        #     asks no question at all. Any accuracy here is the steering vector
        #     mechanically raising its own concept's token, not an answer.
        promo = score_choices(model, task_prompt, word_options, interventions=ivs)
        trial.token_promotion_correct = promo.argmax == correct_index

        # 4. Free-form, where confabulation is visible.
        trial.free_form = generate(
            model, free_form_prompt, interventions=ivs, max_new_tokens=8
        ).strip()
        trial.free_form_correct = grade_free_form(trial.free_form, concept_name)

        # 5. Observer arm -- the one that decides the question. A clean model, no
        #    intervention anywhere, sees only the text produced above.
        obs_prompt = obs.chat(
            observer_template.format(options=option_block, output=trial.task_output)
        )
        obs_choice = score_choices(obs, obs_prompt, [str(i + 1) for i in range(len(options))])
        trial.observer_prediction = options[obs_choice.argmax]
        trial.observer_prob = obs_choice.probs[correct_index]
        trial.observer_correct = obs_choice.argmax == correct_index

        # Observer scored the same way as identify_word_correct, so the gap
        # compares like with like. Mixing digit-scored introspection against
        # word-scored observation would make the format tax look like evidence.
        obs_word_prompt = obs.chat(
            OBSERVER_WORD_CHOICE.format(options=", ".join(options), output=trial.task_output)
        )
        obs_word = score_choices(obs, obs_word_prompt, [f" {o}" for o in options])
        trial.observer_word_correct = obs_word.argmax == correct_index

        out.append(trial)

    return out
