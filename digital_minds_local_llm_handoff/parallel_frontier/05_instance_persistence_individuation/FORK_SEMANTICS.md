# Exact Fork Semantics

At t0 create a clean prompt/cache checkpoint C. From C create two hidden-state siblings C+ and C-.
Apply +v vs -v exactly once, then remove hooks.

Persist resulting cache states K+ and K-. Clone each into measurement descendants so asking a
question cannot contaminate the continuation being measured.

For every later turn:
- **cache mode:** advance K+ and K- with byte-identical forced-neutral text.
- **recompute mode:** rebuild both from the same visible transcript without retained cache.

Primary evidence for instance-level persistence is `delta_cache(k) != 0` while
`delta_recompute(k) ≈ 0`.

Once unconstrained outputs diverge, visible text becomes a mediator. Analyze the pre-divergence
window separately or force identical neutral continuations.
