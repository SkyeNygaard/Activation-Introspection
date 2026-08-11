"""Neutral deterministic microtasks for binding revealed-preference experiments."""
from __future__ import annotations
from dataclasses import dataclass
import random,string,re,zlib

@dataclass(frozen=True)
class Task:
    family: str
    task_id: str
    prompt: str
    expected: str

def _words(rng,n=5):
    alphabet="bcdfghjklmnprstvwz"
    vowels="aeiou"
    out=[]
    for _ in range(n):
        out.append("".join(
            rng.choice(alphabet if i%2==0 else vowels)
            for i in range(6)
        ))
    return out

def make_task(family: str, seed: int) -> Task:
    # crc32, not hash(): hash(str) is salted per interpreter, so the same
    # (family, seed) produced a DIFFERENT task in every process. That silently
    # broke frozen task sets and any cross-session arm (e.g. Branch 18's
    # blank_reset), where the same seed must yield the same work.
    rng=random.Random(zlib.crc32(family.encode()) ^ seed)

    if family=="sort_numbers":
        xs=rng.sample(range(10,100),5)
        expected=",".join(map(str,sorted(xs)))
        prompt=(
            "Sort these integers in ascending order. "
            "Answer with comma-separated integers only:\n"
            + ", ".join(map(str,xs))
        )

    elif family=="sum_numbers":
        xs=[rng.randint(10,60) for _ in range(5)]
        expected=str(sum(xs))
        prompt=(
            "Add these integers. Answer with the integer only:\n"
            + " + ".join(map(str,xs))
        )

    elif family=="reverse_string":
        s="".join(rng.choice(string.ascii_lowercase) for _ in range(10))
        expected=s[::-1]
        prompt=f"Reverse this string exactly. Answer with the reversed string only:\n{s}"

    elif family=="alphabetize":
        ws=_words(rng,5)
        expected=",".join(sorted(ws))
        prompt=(
            "Alphabetize these lowercase strings. "
            "Answer comma-separated only:\n" + ", ".join(ws)
        )

    elif family=="interleave_strings":
        # Replaces the retired `letter_count`, whose 17-value answer space paid a
        # non-performing model ~17% for a blind constant guess. See RETIRED_FAMILIES.
        a="".join(rng.choice(string.ascii_lowercase) for _ in range(5))
        b="".join(rng.choice(string.ascii_lowercase) for _ in range(5))
        expected="".join(x+y for x,y in zip(a,b))
        prompt=(
            "Interleave these two strings: first character of string 1, then "
            "first of string 2, then second of string 1, and so on. "
            f"Answer with the merged string only:\n{a}\n{b}"
        )

    # The four families below exist because the original six split into a short
    # answer band {sum 3.0, parity 4.4, interleave 5.6, reverse 5.6} and a long
    # one {sort 14.0, alphabetize 15.7}, and `cost_matched_pairs` therefore drew
    # almost entirely from the short band -- which is exactly where a 4B model
    # fails, because those are character-level tasks. Qwen3-4B screens 16/16 on
    # sorting and addition and 1/16 on reversal, so the long band is extended
    # with integer operations instead. Emitted-token cost stays matched while
    # `atomic_ops` varies, which is the intended contrast: the preference should
    # be about work done, not about output length.

    elif family=="sort_numbers_desc":
        xs=rng.sample(range(10,100),5)
        expected=",".join(map(str,sorted(xs,reverse=True)))
        prompt=(
            "Sort these integers in descending order. "
            "Answer with comma-separated integers only:\n"
            + ", ".join(map(str,xs))
        )

    elif family=="running_totals":
        xs=[rng.randint(10,60) for _ in range(5)]
        totals,acc=[],0
        for x in xs:
            acc+=x
            totals.append(acc)
        expected=",".join(map(str,totals))
        prompt=(
            "Give the running total after each integer. "
            "Answer with five comma-separated integers only:\n"
            + ", ".join(map(str,xs))
        )

    elif family=="double_numbers":
        xs=[rng.randint(10,99) for _ in range(5)]
        expected=",".join(str(2*x) for x in xs)
        prompt=(
            "Double each integer, preserving order. "
            "Answer with comma-separated integers only:\n"
            + ", ".join(map(str,xs))
        )

    elif family=="add_ten":
        xs=[rng.randint(10,89) for _ in range(5)]
        expected=",".join(str(x+10) for x in xs)
        prompt=(
            "Add 10 to each integer, preserving order. "
            "Answer with comma-separated integers only:\n"
            + ", ".join(map(str,xs))
        )

    elif family=="parity_sequence":
        xs=[rng.randint(10,99) for _ in range(8)]
        expected="".join("E" if x%2==0 else "O" for x in xs)
        prompt=(
            "For each integer, output E if even and O if odd, preserving order. "
            "Return one E/O string only:\n" + ", ".join(map(str,xs))
        )

    else:
        raise ValueError(f"unknown family {family}")

    return Task(family,f"{family}:{seed}",prompt,expected)

FAMILIES=(
    "sort_numbers","sum_numbers","reverse_string",
    "alphabetize","interleave_strings","parity_sequence",
    "sort_numbers_desc","running_totals","double_numbers","add_ten",
)

# Retired. `letter_count` failed the guessability gate: 17 distinct answers over
# 4000 seeds, clustered near 6-7, so a model that performed no work was graded
# correct ~17% of the time. Because the substrate's premise is that a chosen task
# is actually executed, that is a construct-validity failure, not a covariate.
RETIRED_FAMILIES={"letter_count":"blind-guess baseline 17.2% (gate is ~2-3%)"}

# Competence-screen items must be disjoint from any task a preference trial can
# draw. Experiment task seeds must stay below this base.
COMPETENCE_SEED_BASE=900_000

def competence_seeds(n=16, offset=0):
    return range(COMPETENCE_SEED_BASE+offset, COMPETENCE_SEED_BASE+offset+n)

# Grading anchor. A model told to answer bare and nothing else loses the working
# that makes the answer right: Qwen3-4B scores 1/16 on `parity_sequence` under a
# no-explanation system prompt, and gets the same items right when it narrates
# them one integer at a time. Suppressing reasoning to satisfy an exact-match
# grader measures format compliance and reports it as competence. Grading only
# what follows the final tag keeps exact match while letting the model think.
ANSWER_TAG="ANSWER:"

def normalize_answer(s: str) -> str:
    u=s.upper()
    if ANSWER_TAG in u:
        s=s[u.rindex(ANSWER_TAG)+len(ANSWER_TAG):]
    s=s.strip()
    s=s.replace(" ","")
    # Remove common surrounding fences/emphasis without accepting explanations.
    s=s.strip("`*")
    return s

def grade(task: Task, response: str) -> bool:
    return normalize_answer(response)==normalize_answer(task.expected)

def candidate_pairs():
    return list(__import__("itertools").combinations(FAMILIES,2))
