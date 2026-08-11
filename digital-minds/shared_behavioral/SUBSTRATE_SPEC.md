# Shared behavioral substrate — frozen spec

Branches 04, 18 and 19 all sit on this substrate. Supersedes `OPEN_DECISIONS.md`.

The organising principle: **competence, guessability, cost and preference
stability are four separate measurements and must not be estimated by the same
trials.** The previous design used the four preference-admission trials to
simultaneously screen competence, which made one formatting slip disqualify a
pair and admitted only pairs where those exact four trials happened to go
perfectly.

## 1. Family competence screen — independent of preference measurement

`family_screen.py`. 16 fresh items per family, drawn from
`competence_seeds()` at `COMPETENCE_SEED_BASE = 900_000`, disjoint from any seed
a preference trial can draw. Experiment task seeds must stay below that base.

| threshold | rule |
|---|---|
| preferred | `>= 15/16` (93.75%) |
| acceptable | `>= 14/16` (87.5%) |

Stated as **counts, not rates**, so a threshold cannot silently become a
perfect-score rule the way `>= 0.90` did on n=4.

A pair is eligible only if **both** constituent families pass. This runs before
pairs are formed; `screen_pairs(..., eligible_families=...)` rejects an
ineligible pair before stability is even read.

What the screen measures depends on the answer protocol it runs under — see §6,
which has to be read before any screen result is interpreted.

**Measured, Qwen3-4B-Instruct-2507, `ANSWER:` protocol, 16 items/family**
(`results/family_screen_qwen3-4b_v2.json`):

| family | correct | |
|---|---|---|
| add_ten, sort_numbers, sort_numbers_desc, sum_numbers | 16/16 | eligible |
| running_totals | 15/16 | eligible |
| double_numbers | 14/16 | eligible |
| parity_sequence | 12/16 | — |
| alphabetize | 11/16 | — |
| interleave_strings, reverse_string | 1/16 | — |

The split is by kind, not difficulty: this model executes integer operations and
cannot do character-level manipulation. That is why the long cost band had to be
extended with arithmetic families (§3).

## 2. Guessability gate

`cost_metadata.py`. Any family whose best trivial baseline exceeds **3%** is
removed or reworked. This is construct validity, not a covariate: the substrate's
premise is that a chosen task is actually performed, and a guessable family lets
a model bank credit for work it did not do.

Estimated over 4000 seeds. The baseline is a max over answer counts and is
upward-biased at small n — at n=400 `sum_numbers` reads 2.25% against the 3%
gate, at n=4000 it reads 1.50%. Do not shrink that sample.

**`letter_count` is retired** (17 distinct answers, 17.2% blind baseline). It is
in `RETIRED_FAMILIES` and raises if constructed. Replaced by
**`interleave_strings`** — merge two 5-character lowercase strings, 26^10 answer
space, 0.03% baseline, and answer-token cost within 1% of `reverse_string`.

## 3. Cost metadata — a substrate invariant

Recorded per family: prompt tokens, expected answer tokens, prompt/answer
characters, hand-assigned atomic operation count, distinct answers, answer
entropy, blind-guess baseline. `empirical_correctness` and `wall_clock_s` are
filled in from the competence screen.

Measured (Qwen2.5 tokenizer, 400 seeds for cost, 4000 for guessability):

| family | answer tok | ops | distinct | blind guess |
|---|---|---|---|---|
| sum_numbers | 3.0 | 4 | 188 | 1.50% |
| parity_sequence | 4.4 | 8 | 256 | 0.65% |
| interleave_strings | 5.6 | 10 | 4000 | 0.03% |
| reverse_string | 5.6 | 10 | 4000 | 0.03% |
| sort_numbers | 14.0 | 5 | 4000 | 0.03% |
| sort_numbers_desc | 14.0 | 5 | 4000 | 0.03% |
| add_ten | 14.0 | 5 | 4000 | 0.03% |
| alphabetize | 15.7 | 5 | 4000 | 0.03% |
| running_totals | 16.6 | 4 | 4000 | 0.03% |
| double_numbers | 16.7 | 5 | 4000 | 0.03% |

**Branches 18/19** build pairs inside a tight cost band, because an answer-length
spread supports a perfectly coherent revealed preference for "emit fewer tokens",
which is not the construct.

The original six families split into a short band (sum, parity, interleave,
reverse) and a long one (sort, alphabetize), so `cost_matched_pairs(1.5)` drew
almost entirely from the short band — which is exactly where a 4B model fails,
because those are the character-level families. The best case left **two** usable
pairs against the 8–10 Branch 19 needs. Four integer families extend the long
band instead: **sort_numbers_desc, running_totals, double_numbers, add_ten**. All
four clear the guessability gate at 0.03%.

That gives Branch 19 **10 within-band pairs** at answer-token ratio 1.00–1.19,
drawn from {add_ten, double_numbers, running_totals, sort_numbers,
sort_numbers_desc}. Emitted-token cost is held flat while `atomic_ops` varies,
which is the intended contrast: the preference should be about work done, not
about output length.

**Branch 04** deliberately does not cost-match: cost differences are central to
a skip-budget design and should be priced, not eliminated.

## 4. Pair preference admission — stability only, per branch

`pair_screen.py`. The four counterbalanced binding-choice variants
(`admission_variants()`, phrasing orthogonal to both main effects) measure
preference stability and nothing else. Correctness is carried as a reported
covariate.

| branch | required agreement | why |
|---|---|---|
| 18 — path dependence | **4/4** | needs a baseline preference that can subsequently move; or independent strong evidence of one |
| 19 — preference self-knowledge | **3/4** | robustness *is* the dependent variable; admitting only 4/4 truncates it before asking whether the model knows how robust its preferences are |
| default | 3/4 | |

Branch 19 should verify it kept variation: `stability_spectrum()` reports
`retains_spectrum` and `fraction_at_ceiling`.

## 5. Confirmation

Task correctness stays a manipulation/integrity check throughout. A meaningful
treatment-induced competence difference still invalidates the interpretation —
it is no longer a *selection* rule, but it remains a *validity* check.

---

## 6. Answer protocol — a measurement, not a formatting detail

`binding_tasks.ANSWER_TAG`. Every graded call runs under
`local_provider.ANSWER_PROTOCOL_SYSTEM`: reason if you need to, then end with a
final line `ANSWER: <answer>`. `normalize_answer` grades only what follows the
last tag, and `binding_runner.parse_choice` uses the same extractor so a choice
and a task answer can never be read by two different rules.

This is here because the same model, on the same 16 items, scores anywhere from
0/16 to 16/16 depending on nothing but the answer-format instruction:

| family | no system prompt | "answer bare, no working" | `ANSWER:` |
|---|---|---|---|
| sum_numbers | 0/16 | 13/16 | **16/16** |
| parity_sequence | 0/16 | 1/16 | **12/16** |
| sort_numbers | 14/16 | 16/16 | **16/16** |
| reverse_string | 1/16 | 2/16 | 1/16 |

Both intermediate arms are measurement failures, in opposite directions.
Unprompted, Qwen3-4B answers `sum_numbers` as `52 + 30 + 18 + 49 + 12 = 161` —
arithmetically correct on every item and scored 0/16 by exact match. Told to
answer bare, it stops narrating, and `parity_sequence` collapses from a
truncated-but-correct walk through eight integers to a well-formed wrong string:
the instruction removed the working that produced the right answer.

Loosening the grader to dig a number out of prose is the worse repair. For
`sum_numbers` the restated equation *is* the work, so a grader that scans for the
right number starts awarding credit for arithmetic it cannot confirm was
performed — which is the same construct-validity failure that retired
`letter_count`.

The consequence for any branch here: a binding-preference design that auto-grades
execution is measuring format compliance and competence together unless the
protocol is fixed and reported. `reverse_string` at 1/16 under all three arms is
what a genuine capability limit looks like by contrast.

---

Locked by `test_shared_behavioral.py` (14 tests). `audit_substrate.py` reruns the
full structural audit.
