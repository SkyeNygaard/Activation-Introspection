# Figure artifact status

`attention_localization_dev.png` is a descriptive DEV-selection figure generated
from the hash-bound localization summary. Stars mark candidates for a disjoint
head screen; they are not confirmed components. That head screen has since run
and stopped — three of the four starred pairs replicated, `query_marker@23` did
not, and six individual components qualified where the protocol allowed 2–4. Read
the figure as the input to a decision that came out negative, not as a route.

`causal_codebook_icl.png` is the current V2 repair-confirmation figure. It is
regenerated from the checksummed raw artifact by `scripts/analyze_codebook_icl.py`;
its intervals exactly enumerate independent bootstrap resamples of the eight
fixed concepts and three fixed carrier strings.

All other current PNGs are **legacy exploratory figures**:

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
