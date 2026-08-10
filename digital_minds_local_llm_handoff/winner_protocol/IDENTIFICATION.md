# Identification Argument — Why Query-Only Is an Exact Causal Control

This is the core design argument and should appear in the eventual report.

## Notation

For one four-demo episode:

- `s = (s1,...,s4)`, with exactly two `+1` and two `-1` hidden demo states;
- `q ∈ {-1,+1}` is the held-out query state;
- `m ∈ {-1,+1}` is the episode's arbitrary state→label mapping;
- encode `Q = +1`, `K = -1`.

The visible label on demonstration `i` is:

`L_i = m s_i`

and the correct held-out label is:

`Y = m q`.

The exact design contains all six balanced demo-state orders, both mappings, and
both query states.

## Query-only arm

In the query-only arm:

- demonstration markers receive **no hidden edits**;
- the held-out marker receives state `q`;
- the model sees the demonstration labels `L`;
- the model does not see `s` or `m`.

Now pair every episode `(s, m, q)` with:

`(-s, -m, q)`.

The paired episode is present because the design enumerates all balanced orders
and both mappings.

Its visible demonstration labels are identical:

`(-m)(-s_i) = m s_i = L_i`.

Its query intervention is also identical because `q` is unchanged.

Its persona, carrier, and all visible text are identical.

But its correct label flips:

`Y' = (-m)q = -Y`.

Therefore **the entire model input and query intervention are identical for the
pair, while the correct Q/K answer is opposite**.

Consequences:

### Accuracy

Any deterministic query-only predictor must give the same prediction to both
members. It therefore gets exactly one of the pair correct.

Across the exact design:

**query-only accuracy = 0.5 by construction.**

This is a design identity, not empirical evidence.

### Conditional probability of the correct Q/K label

Let the model assign conditional probabilities `(p_Q, p_K)` to Q/K for the
identical paired input, with `p_Q + p_K = 1`.

One member's correct probability is `p_Q`, the other's is `p_K`.

Their mean is exactly:

`(p_Q + p_K) / 2 = 0.5`.

So:

**query-only mean p(correct) = 0.5 by construction too**, assuming the stored
quantity is Q/K-renormalized probability and the paired forwards are numerically
identical.

This makes `target - query_only` especially interpretable.

## Target arm

The target arm uses the same visible prompts and the same query intervention, but
also applies the four hidden demo edits.

For the paired episodes above, those demo edits flip from `s` to `-s`.

Thus the only information that differs between pair members is **causally hidden
in the demonstration activations**.

The correct label also flips.

If target performance rises above query-only, the model must be sensitive to
information carried by the hidden demonstration states (or to some unintended
implementation side channel that covaries with those states).

This is why provenance tests and exact marker-position validation still matter.

## What this identifies

A positive target-minus-query-only contrast supports:

> The model used information introduced through causally varied hidden
> demonstration states to solve an episode-remapped opaque-label task.

It does **not** establish:

- privileged introspection;
- semantic understanding of `vGOLD`;
- consciousness;
- welfare experience;
- direction specificity.

Those require separate comparisons.

## Why persona is compatible with the identification

Persona is held fixed within each exact pair.

The symmetry therefore holds separately for:
- neutral persona;
- upbeat persona;
- downbeat persona.

A persona can shift the model's general Q/K bias and query-only still averages
to 0.5 over the exact pair because the correct label flips while its complete
input remains identical.

The primary robustness question is therefore not “does persona change raw Q/K
bias?” It is:

> Does the **target advantage over the exact query-only identity** survive under
> each persona?

## Random / shuffled controls

Target > query-only establishes hidden-demo use.

It does not establish that the welfare direction is special, because arbitrary
directions may also be learnable through the same channel.

Random or coordinate-shuffled directions answer a different question:

> Is this direction more reportable than matched generic perturbations?

Keep these claims separate.
