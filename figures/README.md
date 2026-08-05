# Figure artifact status

All current PNGs are **legacy exploratory figures**:

- `ift_vs_probe.png` is retracted because its axes use mismatched injection
  sites and its error bars use invalid pseudo-replication. The generating entry
  point now fails loudly.
- `introspector_vs_observer.png` uses a clean, different-transcript observer and
  cannot identify privileged access.
- `matched_kl_gap.png` is a coarse sensitivity diagnostic, not a damage-matched
  causal comparison.
- `detection_by_layer.png` retains the legacy repeated-cell uncertainty model.

They remain only to make the correction trail auditable. Replacement figures
must be generated from checksummed raw artifacts and carry descriptive or
confirmatory status in their caption.
