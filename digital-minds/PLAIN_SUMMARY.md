# What we found, in plain language

Written for someone with no background in this project.

## The question

Language models have internal states — patterns of numbers inside the network
that change as the model reads and writes. A natural question is whether a model
can *tell you about its own internal state*. Not "describe your feelings" in a
vague way, but something checkable: we put a specific state into the model
ourselves, so we know exactly what is in there, and then we ask the model to
report it.

If a model could do that reliably, it would be genuinely useful — you could ask a
model what it is actually doing rather than trusting what it says. It also
matters for questions about AI welfare, because "the model says it feels fine"
is worth very little if the model has no real access to its own internal state.

## How the test works

Researchers have published a direction — think of it as a dial inside the
model — that makes the model behave as though things are going well or badly for
it. Turning the dial one way pushes toward "well", the other way toward "badly".

The experiment:

1. Secretly turn the dial up or down at four points in a conversation, and each
   time *show* the model an arbitrary label ("Q" or "K") for what we just did.
   The labels are reshuffled every time, so there is nothing to memorize.
2. Then set the dial secretly a fifth time and ask: which label applies now?
3. The text the model sees is *identical* either way. The only difference is the
   hidden internal state.

If the model can read its own state, it answers correctly. If not, it guesses.
Chance is 50%.

## What we found

**First, most of the setup was broken, and not in the way anyone expected.**

The published dial had a documented setting for *where* in the network to apply
it. That setting was wrong for our copy of it — at the documented location the
dial did nothing at all. We tested all 36 possible locations and found the dial
works powerfully at a different one. An earlier note in the project had guessed
that location correctly, and had then been "corrected" on the basis of reading
documentation. The original guess was right.

**Second, we were writing the hidden state in a place nothing reads.**

The design put the hidden state on a single word of the conversation. We measured
how much that actually affects anything: essentially nothing. We turned the dial
up to *four times* the strength of the entire signal at that point in the network
— an absurd, model-breaking amount — and the model's behaviour did not shift at
all. The state was being written somewhere the rest of the model never looks.

This is worth stating plainly because every result the project had recorded up to
this point was simulated, and all of them assumed this worked.

**Third, we fixed it, and the model still cannot report its state.**

We widened where the state gets written from one word to about fifty, which we
measured to be strong enough to matter. Then we ran the real test:

- The model scored **exactly 50%** — pure chance.
- Its answers were perfectly well-formed, so it was not confused or broken.
- When we ran the same task with the state written in *plain text* instead of
  hidden inside the network, the model scored **48 out of 48**. So the task
  itself is easy for it.

**Fourth — and this is the part that makes the result meaningful — the
information was there.**

We looked directly inside the network at the exact spot the model produces its
answer from, and asked whether *we* could tell which way the dial had been set.
We could, about 87% of the time, and a statistical test says that is not luck
(roughly a 1-in-2000 chance of arising randomly).

We then checked the obvious objection: maybe we were just detecting the dial we
turned, still sitting there untouched, rather than anything the model did with
it. So we mathematically removed the dial's own direction and looked again. The
signal survived almost unchanged (85%), and it points in a direction almost
unrelated to the dial itself. So it is something the model computed, not our own
fingerprint.

## What this means

Put together:

- the information about the model's internal state **is present**, right where
  the answer gets made;
- an outside observer **can read it**;
- the model can do the labelling task perfectly **when the state is visible**;
- and the model reports its own hidden state **at chance**.

The plain reading: for this kind of state, the model does not have introspective
access to something an outside observer can read straight out of its own
internals. That is a negative result, but a clean one — negative results are only
worth anything when you can show the equipment was working, and here we can show
it four different ways.

**Limits, stated honestly.** This is one model, one dial, one location, one task
format, and a simple kind of readout. It does not show that models can never
introspect. It shows this particular thing does not work, and — more usefully —
it shows how to check whether your equipment is actually working before you
believe a result.

## The second experiment, briefly

Separately we tested whether a model's *choices* are economically consistent when
they have real consequences. We gave it a budget and two kinds of chore, made it
pick a combination, and then made it actually do the work it had chosen. Its
sorting work was checked against the right answers, so it could not fake it.

It handled this perfectly — 48 out of 48 valid choices, all work genuinely done.
That is a useful capability result on its own.

But when we measured how *consistent* its choices were, we hit a wall that is
worth knowing about: we simulated a coin-flipping agent making random choices,
and it scored **better** than the model on three of four conditions. The test
simply cannot tell a thoughtful chooser from a random one at this size. We swept
every setting we could — more budgets, bigger budgets, more items — and random
choice kept scoring perfectly some of the time.

So we are reporting the capability result and explicitly *not* reporting the
consistency numbers, because they do not mean anything yet. The reason is
structural: making choices have real consequences is what makes this experiment
interesting, and it is also what limits how many choices you can afford to
collect. Fixing it needs a redesign, not a bigger run.
