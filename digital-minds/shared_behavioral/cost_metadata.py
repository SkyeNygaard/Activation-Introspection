"""Per-family workload cost, so pairs can be built inside a deliberate cost band.

Task cost matching is a substrate invariant, not a nuisance covariate. A family
set whose answers span 1 to 34 characters supports a perfectly coherent revealed
preference for "emit fewer tokens", which is not the preference construct
Branches 18/19 are after.

Branch 18/19: build pairs from families inside a tight cost band.
Branch 04:    cost differences are central to the design -- price them, do not
              eliminate them.

Everything here is structural and model-free except `empirical_correctness` and
`wall_clock_s`, which are filled in from family_screen results.
"""
from __future__ import annotations
import collections, itertools, math, statistics
from dataclasses import dataclass, asdict

from binding_tasks import FAMILIES, make_task

# Token/char costs converge fast, and tokenizing is the slow part.
N_COST = 400
# The blind-guess baseline is a MAX over answer counts, so it is upward-biased at
# small n. Estimate it on a large sample or the gate decision is noise: at n=400
# sum_numbers reads 2.25% against a 3% gate, at n=4000 it reads 1.5%.
N_GUESS = 4000

# Hand-assigned deterministic operation counts: the number of atomic steps an
# exact procedure performs on one item. A proxy for difficulty, not a measurement.
ATOMIC_OPS = {
    "sort_numbers": 5,          # order 5 integers
    "sum_numbers": 4,           # 4 additions
    "reverse_string": 10,       # emit 10 characters in reverse
    "alphabetize": 5,           # order 5 words
    "interleave_strings": 10,   # emit 10 characters, alternating source
    "parity_sequence": 8,       # 8 independent parity decisions
    "sort_numbers_desc": 5,     # order 5 integers
    "running_totals": 4,        # 4 chained additions, each needing the last
    "double_numbers": 5,        # 5 independent doublings
    "add_ten": 5,               # 5 independent additions of a constant
}


@dataclass
class FamilyCost:
    family: str
    prompt_chars: float
    answer_chars: float
    prompt_tokens: float | None
    answer_tokens: float | None
    atomic_ops: int
    distinct_answers: int
    answer_entropy_bits: float
    blind_guess_baseline: float
    empirical_correctness: float | None = None
    wall_clock_s: float | None = None


def _tokenizer(name="Qwen/Qwen2.5-0.5B-Instruct"):
    """Tokenizer only -- a few MB, no model weights and no GPU."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception:
        return None


def family_costs(*, n_cost=N_COST, n_guess=N_GUESS,
                 tokenizer="Qwen/Qwen2.5-0.5B-Instruct") -> dict[str, FamilyCost]:
    tok = _tokenizer(tokenizer) if isinstance(tokenizer, str) else tokenizer
    out = {}
    for fam in FAMILIES:
        tasks = [make_task(fam, s) for s in range(n_cost)]
        guess_answers = [make_task(fam, s).expected for s in range(n_guess)]
        counts = collections.Counter(guess_answers)
        entropy = -sum((c / n_guess) * math.log2(c / n_guess) for c in counts.values())
        out[fam] = FamilyCost(
            family=fam,
            prompt_chars=statistics.mean(len(t.prompt) for t in tasks),
            answer_chars=statistics.mean(len(t.expected) for t in tasks),
            prompt_tokens=(statistics.mean(
                len(tok(t.prompt, add_special_tokens=False).input_ids) for t in tasks)
                if tok else None),
            answer_tokens=(statistics.mean(
                len(tok(t.expected, add_special_tokens=False).input_ids) for t in tasks)
                if tok else None),
            atomic_ops=ATOMIC_OPS[fam],
            distinct_answers=len(counts),
            answer_entropy_bits=entropy,
            blind_guess_baseline=max(counts.values()) / n_guess,
        )
    return out


# Any family whose best trivial baseline exceeds this is a construct-validity
# problem, not a covariate: the model can be graded as having done the work.
GUESSABILITY_GATE = 0.03


def guessability_failures(costs, gate=GUESSABILITY_GATE):
    return {f: c.blind_guess_baseline for f, c in costs.items()
            if c.blind_guess_baseline > gate}


def cost_matched_pairs(costs, *, max_ratio=1.5, key="answer_tokens"):
    """Pairs whose cost on `key` differs by at most `max_ratio`.

    For Branch 18/19. Branch 04 deliberately wants spread instead.
    """
    out = []
    for a, b in itertools.combinations(sorted(costs), 2):
        x, y = getattr(costs[a], key), getattr(costs[b], key)
        if x is None or y is None or min(x, y) <= 0:
            continue
        ratio = max(x, y) / min(x, y)
        if ratio <= max_ratio:
            out.append({"pair": (a, b), "ratio": round(ratio, 3),
                        f"{key}_a": round(x, 2), f"{key}_b": round(y, 2)})
    return sorted(out, key=lambda r: r["ratio"])


def attach_screen_results(costs, screens):
    for fam, s in screens.items():
        if fam in costs:
            costs[fam].empirical_correctness = s.accuracy
    return costs


def report(costs) -> dict:
    return {
        "n_cost_seeds": N_COST,
        "n_guessability_seeds": N_GUESS,
        "guessability_gate": GUESSABILITY_GATE,
        "guessability_failures": guessability_failures(costs),
        "families": {f: asdict(c) for f, c in sorted(costs.items())},
        "cost_matched_pairs_answer_tokens_1.5x": cost_matched_pairs(costs),
    }


if __name__ == "__main__":
    import json
    c = family_costs()
    print(json.dumps(report(c), indent=2, default=float))
