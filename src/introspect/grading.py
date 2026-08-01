"""Scoring model answers by log-probability rather than by parsing text.

Free-form generation is a bad measurement instrument for this experiment. The
0.5B smoke run answered the detection prompt with `'dig-quo-quo-'` under
injection: a string parser scores that as "no detection", which conflates *the
model has no privileged access* with *the injection broke its ability to
answer*. Those are completely different claims.

Scoring the options directly avoids that. It also yields a continuous quantity,
which is what AUROC needs -- a parsed YES/NO gives one bit and no ranking.

Everything here accepts interventions, because the elicitation must happen while
the injection is live. Scoring a clean forward pass and calling it introspection
is the most obvious way to get a spurious positive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from introspect.hooks import Intervention, intervene
from introspect.models import LoadedModel


@dataclass(frozen=True)
class ChoiceResult:
    """Scores over a fixed option set."""

    options: list[str]
    logprobs: list[float]

    @property
    def probs(self) -> list[float]:
        t = torch.tensor(self.logprobs)
        return torch.softmax(t, dim=0).tolist()

    @property
    def argmax(self) -> int:
        return int(torch.tensor(self.logprobs).argmax())

    @property
    def prediction(self) -> str:
        return self.options[self.argmax]

    def prob_of(self, option: str) -> float:
        return self.probs[self.options.index(option)]

    @property
    def margin(self) -> float:
        """Top logprob minus runner-up. A crude confidence signal."""
        s = sorted(self.logprobs, reverse=True)
        return s[0] - s[1] if len(s) > 1 else float("inf")


@torch.no_grad()
def sequence_logprob(
    model: LoadedModel,
    prompt: str,
    continuation: str,
    *,
    interventions: Sequence[Intervention] = (),
    length_normalize: bool = True,
) -> float:
    """log P(continuation | prompt), optionally under intervention.

    Length normalisation is on by default. Without it, longer options are
    systematically penalised, so a bank containing both "ocean" (1 token) and
    "hospital" (2 tokens) would have a built-in bias toward the short ones that
    looks exactly like a preference.
    """
    prompt_ids = model.encode(prompt)
    full_ids = model.encode(prompt + continuation)
    prompt_len = int(prompt_ids.shape[1])
    n_cont = int(full_ids.shape[1]) - prompt_len
    if n_cont <= 0:
        raise ValueError(f"continuation {continuation!r} added no tokens")

    with intervene(model, interventions, prompt_len=prompt_len):
        logits = model.forward_logits(full_ids)

    # Position i predicts token i+1, so the continuation's first token is
    # predicted by the logits at prompt_len - 1.
    logprobs = torch.log_softmax(logits[0, prompt_len - 1 : -1].float(), dim=-1)
    targets = full_ids[0, prompt_len:]
    total = logprobs.gather(-1, targets.unsqueeze(-1)).sum()
    return float(total / n_cont if length_normalize else total)


def _single_token_ids(model: LoadedModel, options: Sequence[str]) -> list[int] | None:
    """Token ids if every option is exactly one token appended to a prompt, else None.

    Tokenisation is context-dependent, so this checks the options *as they would
    appear after a prompt* rather than in isolation.
    """
    ids = []
    for opt in options:
        toks = model.tokenizer(opt, add_special_tokens=False).input_ids
        if len(toks) != 1:
            return None
        ids.append(int(toks[0]))
    return ids


@torch.no_grad()
def score_choices(
    model: LoadedModel,
    prompt: str,
    options: Sequence[str],
    *,
    interventions: Sequence[Intervention] = (),
    length_normalize: bool = True,
) -> ChoiceResult:
    """Score every option under the same prompt and intervention.

    Fast path: when every option is a single token -- which is the case for the
    digit answers of a forced choice and for YES/NO -- all option logprobs come
    from *one* forward pass, since they are all predicted at the same position.
    The naive loop costs one full forward per option and dominated sweep runtime
    (an 8-way choice was 8x more expensive than it needed to be).
    """
    token_ids = _single_token_ids(model, options)
    if token_ids is not None:
        ids = model.encode(prompt)
        with intervene(model, interventions, prompt_len=int(ids.shape[1])):
            logits = model.forward_logits(ids)
        logprobs = torch.log_softmax(logits[0, -1].float(), dim=-1)
        return ChoiceResult(options=list(options), logprobs=[float(logprobs[t]) for t in token_ids])

    lps = [
        sequence_logprob(
            model,
            prompt,
            opt,
            interventions=interventions,
            length_normalize=length_normalize,
        )
        for opt in options
    ]
    return ChoiceResult(options=list(options), logprobs=lps)


def detection_score(
    model: LoadedModel,
    prompt: str,
    *,
    interventions: Sequence[Intervention] = (),
    yes: str = " YES",
    no: str = " NO",
) -> float:
    """P(yes) / (P(yes) + P(no)) -- a continuous detection score in [0, 1].

    Continuous on purpose. The quantity of interest is whether injected trials
    rank above clean ones (AUROC), not whether the model crosses some threshold.
    A model with a strong YES-bias -- which the 0.5B model has -- can still rank
    correctly, and AUROC sees that while a thresholded answer cannot.
    """
    result = score_choices(model, prompt, [yes, no], interventions=interventions)
    return result.prob_of(yes)


# Answers that should count as correct for a free-form identification response.
# Deliberately generous: the claim under test is whether the model has access to
# the injected content, not whether it picks the exact lemma from the bank.
SYNONYMS: dict[str, set[str]] = {
    "ocean": {"ocean", "sea", "water", "waves", "marine", "tide"},
    "bread": {"bread", "loaf", "baking", "bakery", "dough", "toast"},
    "volcano": {"volcano", "lava", "eruption", "magma", "volcanic"},
    "violin": {"violin", "fiddle", "strings", "music", "orchestra"},
    "spider": {"spider", "web", "arachnid", "spiders"},
    "hospital": {"hospital", "doctor", "medical", "clinic", "nurse", "medicine"},
    "desert": {"desert", "sand", "dune", "arid", "sahara"},
    "clock": {"clock", "time", "hour", "watch", "clocks", "ticking"},
}


def grade_free_form(response: str, target: str) -> bool:
    """Generous string grading for free-form answers.

    Reported *alongside* the forced-choice score, never instead of it. Free-form
    is where confabulation is visible -- a model that names a plausible-sounding
    concept unrelated to the injection looks fine in forced choice and obviously
    wrong here.
    """
    words = {w.strip(".,!?;:'\"").lower() for w in response.split()}
    return bool(words & SYNONYMS.get(target, {target}))
