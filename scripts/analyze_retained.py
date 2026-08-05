"""Analyse a retained-trace run: storage, verbalization, and functional use.

Three rungs are measured on the *same* retained carrier state, which is what the
retracted `r = -0.774` comparison failed to do:

1. **storage**    -- can a probe trained on ordinary text mentioning the concept
                     recover it from the retained carrier activation?
2. **use**        -- can the model bind that state to a codebook label sampled
                     after the edit was removed?

Where the ladder breaks is the result. Storage without use localizes the failure
to readout rather than retention; neither is licensed by the other.

Inference resamples the two real sampling dimensions -- concept and carrier
family -- as clusters. Codebook permutations are nuisance marginalization and are
never resampled as if independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from introspect import concepts as concept_mod
from introspect import models, probe, retained
from introspect.hooks import capture

CONTROLS = ["clean", "sham", "random", "shuffled"]

# Only these arms can *fail*. `clean` and `sham` run one forward per (carrier,
# codebook) and score it against all eight concepts; with cyclic codebooks
# exactly one of the eight is correct, so their accuracy is 1/n by arithmetic
# whatever the model does. They are leakage diagnostics for the pipeline, not
# evidence about the concept. A contrast quoted against them would be quoted
# against a constant, so the reported effect uses the strongest arm that carries
# a per-concept edit. `shuffled` leads so that ties resolve to it: it preserves
# the reference vector's norm and coordinate distribution and destroys only the
# direction, which is the tighter of the two.
CONCEPT_VARYING = ["shuffled", "random"]


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    stat: Any,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Resample (concept, carrier) clusters, not individual trials.

    Each concept x carrier cell contributes many rows that share a prompt, a
    concept vector, and a carrier state. Treating those as independent is what
    produced the over-narrow legacy intervals.
    """
    by_cluster: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cluster[(r["concept"], r["carrier_id"])].append(r)
    clusters = list(by_cluster.values())
    rng = np.random.default_rng(seed)

    point = stat([r for c in clusters for r in c])
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(clusters), len(clusters))
        sample = [r for i in idx for r in clusters[i]]
        val = stat(sample)
        if not np.isnan(val):
            draws.append(val)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def mean_correct(rows: list[dict[str, Any]]) -> float:
    return float(np.mean([r["correct"] for r in rows])) if rows else float("nan")


def contrast(rows: list[dict[str, Any]], arm_a: str, arm_b: str) -> Any:
    """Paired difference in accuracy between two arms, matched within cluster."""

    def stat(sample: list[dict[str, Any]]) -> float:
        a = [r["correct"] for r in sample if r["arm"] == arm_a]
        b = [r["correct"] for r in sample if r["arm"] == arm_b]
        if not a or not b:
            return float("nan")
        return float(np.mean(a) - np.mean(b))

    return stat


def usability_table(rows: list[dict[str, Any]], sesoi: float) -> None:
    cells = sorted({(r["inject_layer"], r["strength"]) for r in rows if r["arm"] == "target"})
    print("\n=== USE: post-codebook label accuracy ===")
    ceiling = mean_correct([r for r in rows if r["arm"] == "natural"])
    if not np.isnan(ceiling):
        print(f"ceiling (concept stated in plain text, same pipeline): {ceiling:.3f}")
    print(
        "  L = injection layer | a = strength | kl = how much the edit disturbed "
        "the carrier\n"
        "  fmt = fraction of trials still emitting a label | tgt|fmt = target "
        "accuracy on those trials only"
    )
    print(
        f"{'L':>3} {'a':>5} {'kl':>7} {'fmt':>5} {'tgt|fmt':>7} {'target':>7} "
        f"{'clean':>7} {'sham':>7} {'random':>7} {'shuf':>7}   target-vs-best-control 95% CI"
    )
    for layer, strength in cells:
        sub = [
            r
            for r in rows
            if r["inject_layer"] == layer
            and (r["strength"] == strength or r["arm"] in ("clean", "sham"))
        ]
        tgt = [r for r in sub if r["arm"] == "target"]
        accs = {a: mean_correct([r for r in sub if r["arm"] == a]) for a in ["target", *CONTROLS]}
        kl = float(np.mean([r["carrier_kl"] for r in tgt]))
        fmt = float(np.mean([r["format_ok"] for r in tgt]))
        # Strengths that maximise accuracy also damage formatting. If the effect
        # only exists on trials where the model no longer emits a label as its
        # unrestricted argmax, the forced-choice score is measuring damage.
        fmt_only = mean_correct([r for r in tgt if r["format_ok"]])

        # The weakest contrast is the honest one: a target arm must beat the
        # *strongest* control, not the average of them -- and the control has to
        # be one that could have come out otherwise. See CONCEPT_VARYING.
        best_ctrl = max(CONCEPT_VARYING, key=lambda a: accs[a])
        point, lo, hi = cluster_bootstrap(sub, contrast(sub, "target", best_ctrl))
        verdict = ""
        if lo > sesoi:
            verdict = "  <-- exceeds SESOI"
        elif -sesoi < lo and hi < sesoi:
            verdict = "  (equivalent)"
        print(
            f"{layer:>3} {strength:>5g} {kl:>7.2f} {fmt:>5.2f} {fmt_only:>7.3f} "
            f"{accs['target']:>7.3f} "
            + " ".join(f"{accs[a]:>7.3f}" for a in CONTROLS)
            + f"   {point:+.3f} [{lo:+.3f},{hi:+.3f}] vs {best_ctrl}{verdict}"
        )


def damage_table(rows: list[dict[str, Any]]) -> None:
    """Carrier KL and format integrity by arm.

    A control only controls for damage if it inflicts comparable damage. If the
    target arm perturbs the carrier far less than `random`/`shuffled`, the
    contrast is confounded with how much the edit broke the model, in whichever
    direction happens to flatter the hypothesis.
    """
    print("\n=== DAMAGE MATCHING: carrier KL / format integrity by arm ===")
    cells = sorted({(r["inject_layer"], r["strength"]) for r in rows if r["arm"] == "target"})
    print(f"{'L':>3} {'a':>5} " + " ".join(f"{a:>16}" for a in ["target", "random", "shuffled"]))
    for layer, strength in cells:
        sub = [r for r in rows if r["inject_layer"] == layer and r["strength"] == strength]
        parts = []
        for arm in ("target", "random", "shuffled"):
            a_rows = [r for r in sub if r["arm"] == arm]
            if not a_rows:
                parts.append(f"{'-':>16}")
                continue
            kl = float(np.mean([r["carrier_kl"] for r in a_rows]))
            fmt = float(np.mean([r["format_ok"] for r in a_rows]))
            parts.append(f"{kl:>8.2f}/{fmt:<7.2f}")
        print(f"{layer:>3} {strength:>5g} " + " ".join(parts))


def kl_matched_profile(rows: list[dict[str, Any]], target_kl: float) -> None:
    """Depth profile at matched carrier damage.

    Comparing layers at a fixed strength confounds depth with damage: the same
    alpha produces a different carrier KL at each site. This instead selects,
    per layer, the strength whose target-arm KL is closest to a common band, so
    the remaining difference across depth is not just "the early edit hurt more".
    """
    print(f"\n=== DEPTH PROFILE at matched carrier KL (~{target_kl:g}) ===")
    print(f"{'L':>3} {'a':>5} {'kl':>7} {'target':>8} {'best ctrl':>10}   effect 95% CI")
    layers = sorted({r["inject_layer"] for r in rows if r["arm"] == "target"})
    for layer in layers:
        opts = sorted({r["strength"] for r in rows if r["inject_layer"] == layer and r["strength"]})
        if not opts:
            continue
        kls = {
            s: float(
                np.mean(
                    [
                        r["carrier_kl"]
                        for r in rows
                        if r["inject_layer"] == layer
                        and r["strength"] == s
                        and r["arm"] == "target"
                    ]
                )
            )
            for s in opts
        }
        best_s = min(kls, key=lambda s: abs(kls[s] - target_kl))
        sub = [
            r
            for r in rows
            if r["inject_layer"] == layer
            and (r["strength"] == best_s or r["arm"] in ("clean", "sham"))
        ]
        accs = {a: mean_correct([r for r in sub if r["arm"] == a]) for a in ["target", *CONTROLS]}
        best_ctrl = max(CONCEPT_VARYING, key=lambda a: accs[a])
        point, lo, hi = cluster_bootstrap(sub, contrast(sub, "target", best_ctrl))
        print(
            f"{layer:>3} {best_s:>5g} {kls[best_s]:>7.2f} {accs['target']:>8.3f} "
            f"{accs[best_ctrl]:>10.3f}   {point:+.3f} [{lo:+.3f},{hi:+.3f}]"
        )


def per_concept_table(rows: list[dict[str, Any]], layer: int, strength: float) -> None:
    """Per-concept accuracy at one cell.

    A pooled mean of 0.5 is a different claim depending on whether every concept
    sits near 0.5 or two concepts sit at 1.0 and six at chance. The second
    pattern is a lexical quirk of those two vectors, not a general capability.
    """
    tgt = [
        r
        for r in rows
        if r["arm"] == "target" and r["inject_layer"] == layer and r["strength"] == strength
    ]
    if not tgt:
        return
    print(f"\n=== SPECIFICITY: per-concept accuracy at L{layer}, a={strength:g} ===")
    by_concept = defaultdict(list)
    for r in tgt:
        by_concept[r["concept"]].append(r["correct"])
    for concept in sorted(by_concept, key=lambda c: -float(np.mean(by_concept[c]))):
        vals = by_concept[concept]
        bar = "#" * round(float(np.mean(vals)) * 40)
        print(f"  {concept:>10} {float(np.mean(vals)):.3f} n={len(vals):<4} {bar}")
    n_above = sum(1 for c in by_concept if float(np.mean(by_concept[c])) > 0.25)
    print(f"  concepts above 2x chance: {n_above}/{len(by_concept)}")


def propagation_control(
    m: Any,
    acts_path: Path,
    concepts: list[str],
    readout: int,
    strength: float,
) -> dict[int, tuple[float, float, float]]:
    """Is storage the model retaining a trace, or just vector addition?

    ``probe.py`` documents the trap this has to clear: the residual stream is
    additive, so a probe that recovers an injected direction may have recovered
    only what we added. Training the probe on natural text narrows that but does
    not close it -- the concept vector could simply be geometrically aligned with
    the natural-text boundary already.

    The control skips the model. It takes the CLEAN carrier state at ``readout``
    and adds the same delta the intervention would have added at the injection
    layer, with no forward computation between the two. If the probe scores that
    as highly as the real target arm, the storage number is arithmetic. If it
    does not, the alignment had to be produced by the intervening blocks, which
    is a claim about the model.

    A third column drops the carrier entirely and probes the delta alone.
    """
    blob = torch.load(acts_path, weights_only=False)
    index, acts = blob["index"], blob["acts"]
    inject_layers = sorted({e["inject_layer"] for e in index})

    x_nat, y_nat, _ = probe.collect_natural(m, concepts, readout)
    scaler = StandardScaler().fit(x_nat)
    clf = LogisticRegression(max_iter=3000).fit(scaler.transform(x_nat), y_nat)

    labels = retained.single_token_labels(m)[: len(concepts)]
    codebook = retained.balanced_codebooks(concepts, labels)[0]
    clean: dict[int, torch.Tensor] = {}
    scale: dict[tuple[int, int], float] = {}
    for carrier_id, carrier_text in enumerate(retained.CARRIER_FAMILIES):
        split = retained.split_prompt(m, carrier_text, codebook)
        with torch.no_grad(), capture(m, list(range(m.n_layers))) as store:
            m.model(split.stage1)
        clean[carrier_id] = store.acts[readout][0][:, -1, :].squeeze(0).float()
        positions = retained.carrier_positions(int(split.stage1.shape[1]))
        for layer in inject_layers:
            # Mirrors hooks.Intervention.apply with normalize=True.
            edited = store.acts[layer][0][:, positions, :]
            scale[(layer, carrier_id)] = strength * float(edited.norm(dim=-1, keepdim=True).mean())

    print(f"\n=== PROPAGATION CONTROL: is storage arithmetic? (readout {readout}) ===")
    print(f"chance = {1 / len(concepts):.3f}")
    print(f"{'inject':>7} {'real':>8} {'synthetic':>10} {'delta alone':>12}")
    out: dict[int, tuple[float, float, float]] = {}
    for layer in inject_layers:
        bank = concept_mod.build_bank(m, layer, list(concepts))
        sel = [e for e in index if e["arm"] == "target" and e["inject_layer"] == layer]
        truth = np.array([concepts.index(e["concept"]) for e in sel])

        def score(feats: np.ndarray, truth: np.ndarray = truth) -> float:
            return float(np.mean(clf.predict(scaler.transform(feats)) == truth))

        real = np.stack([acts[e["row"], readout].float().numpy() for e in sel])
        deltas = [scale[(layer, e["carrier_id"])] * bank[e["concept"]].unit() for e in sel]
        pairs = zip(sel, deltas, strict=True)
        synth = np.stack([(clean[e["carrier_id"]] + d).numpy() for e, d in pairs])
        alone = np.stack([d.numpy() for d in deltas])
        out[layer] = (score(real), score(synth), score(alone))
        print(f"{layer:>7} {out[layer][0]:>8.3f} {out[layer][1]:>10.3f} {out[layer][2]:>12.3f}")
    print(
        "  A synthetic column near chance means the intervening blocks, not the\n"
        "  injected vector, produced the alignment. At inject == readout the edit\n"
        "  is captured on the same block it was applied to, so that row is\n"
        "  arithmetic by construction and is not evidence."
    )
    return out


def decodability_table(
    m: Any,
    acts_path: Path,
    concepts: list[str],
    readouts: list[int],
    strength: float | None = None,
) -> dict[tuple[int, int], float]:
    """Transfer probe: trained on natural mentions, tested on the retained state.

    The probe never sees an injected example. It is fit only on ordinary
    sentences that mention each concept, so above-chance transfer means the
    retained carrier state occupies the same representational format the model
    uses for a genuine mention -- not merely that "something was added".

    Two readings are printed per cell, `target/control`. The control column is
    the best of the damage-matched random and shuffled arms; it must sit at
    chance, otherwise the probe is reading injection magnitude rather than
    concept identity. Readout layers *below* the injection site are also a
    negative control: the edit has not happened yet there.

    This does not on its own separate retention from vector addition; see
    ``propagation_control``.
    """
    blob = torch.load(acts_path, weights_only=False)
    index, acts = blob["index"], blob["acts"]
    if strength is not None:
        index = [e for e in index if e.get("strength", strength) == strength]

    out: dict[tuple[int, int], float] = {}
    print("\n=== STORAGE: natural-text probe transferred to the retained carrier ===")
    print(f"chance = {1 / len(concepts):.3f}; cells are target/control by inject layer")
    inject_layers = sorted({e["inject_layer"] for e in index})
    header = "  ".join(f"{'L' + str(i):>11}" for i in inject_layers)
    print(f"{'readout':>8} {'probeCV':>8}   {header}")

    for r_layer in readouts:
        x_nat, y_nat, g_nat = probe.collect_natural(m, concepts, r_layer)
        # Grouped CV holds out whole sentence frames; it validates that the
        # probe learned the concept rather than the template. If this is at
        # chance the transfer numbers below are meaningless.
        cv = float(np.mean(probe.fit_probe_grouped(x_nat, y_nat, g_nat)))
        scaler = StandardScaler().fit(x_nat)
        clf = LogisticRegression(max_iter=3000).fit(scaler.transform(x_nat), y_nat)

        def acc_for(
            arm: str,
            i_layer: int,
            *,
            r_layer: int = r_layer,
            clf: Any = clf,
            scaler: Any = scaler,
        ) -> float:
            sel = [e for e in index if e["arm"] == arm and e["inject_layer"] == i_layer]
            if not sel:
                return float("nan")
            feats = np.stack([acts[e["row"], r_layer].float().numpy() for e in sel])
            truth = np.array([concepts.index(e["concept"]) for e in sel])
            return float(np.mean(clf.predict(scaler.transform(feats)) == truth))

        cells = []
        for i_layer in inject_layers:
            tgt = acc_for("target", i_layer)
            ctrl = max(acc_for("random", i_layer), acc_for("shuffled", i_layer))
            out[(i_layer, r_layer)] = tgt
            marker = "*" if r_layer < i_layer else " "
            cells.append(f"{tgt:.3f}/{ctrl:.3f}{marker}")
        print(f"{r_layer:>8} {cv:>8.3f}   " + "  ".join(f"{c:>11}" for c in cells))
    print("  * readout below the injection site: negative control, expect chance")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--sesoi", type=float, default=0.05)
    ap.add_argument("--readouts", default="", help="comma-separated; default = inject layers")
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--cell", default="", help="'layer,strength' for the per-concept breakdown")
    ap.add_argument("--match-kl", type=float, default=1.0, help="carrier-KL band for the profile")
    args = ap.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    rows = load_rows(Path(args.raw))
    print(
        f"{len(rows)} rows | model={summary['model']} split={summary['split']} "
        f"chance={summary['chance']:.3f} SESOI={args.sesoi}"
    )

    usability_table(rows, args.sesoi)
    damage_table(rows)
    if len({r["strength"] for r in rows if r["arm"] == "target"}) > 1:
        kl_matched_profile(rows, args.match_kl)

    if args.cell:
        layer_s, strength_s = args.cell.split(",")
        per_concept_table(rows, int(layer_s), float(strength_s))

    if not args.skip_probe and summary.get("activations"):
        acts_path = Path(args.raw).parent / summary["activations"]
        readouts = (
            [int(x) for x in args.readouts.split(",")] if args.readouts else summary["layers"]
        )
        strength = float(args.cell.split(",")[1]) if args.cell else None
        m = models.load(summary["model"])
        try:
            decodability_table(m, acts_path, summary["concepts"], readouts, strength)
            propagation_control(m, acts_path, summary["concepts"], max(readouts), strength or 1.0)
        finally:
            m.free()


if __name__ == "__main__":
    main()
