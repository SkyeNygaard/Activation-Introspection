"""Where does the answer become causally load-bearing at the final token?

Post-hoc diagnostic, DEV bank only, no reporting claim. The frozen arithmetic
pilot stopped because a single-position transplant at layers 9, 21 and 26 left
the ordinary answer unchanged. ``notes/09`` names the reopen condition for this
family: localize a site whose cross-patch actually changes the answer. Two
measurements over every layer:

* **transplant** -- replace the last pre-answer residual with the twin's and
  score recovery of the donor's answer, exactly as the frozen gate does;
* **logit lens** -- read the clean residual through the final norm and
  unembedding, so a layer where the answer is already legible but not
  transplantable separates downstream recomputation from an absent state.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import torch

# Same directory, so this needs no path surgery; it also sets the MPS watermark.
from run_natural_state import (
    MODEL,
    _load,
    _normalized_recovery,
    _sha256,
    _single_continuation_id,
)
from torch import Tensor

from introspect import models
from introspect.hooks import capture
from introspect.natural_state import ARITH_DEV, ArithTask, patch_residuals
from introspect.preflight import check as preflight_check

ROOT = Path(__file__).resolve().parents[1]
ANSWER_PREFIX = "Answer: "
OUT = ROOT / "results/natural_state_arith_site_diagnostic_v1.json"


class Clean:
    """One solved problem: its prompt, its answer pair, and every residual."""

    def __init__(self, model: models.LoadedModel, task: ArithTask, sign: int) -> None:
        self.prompt = model.chat(task.render_user(sign), assistant_prefix=ANSWER_PREFIX)
        self.input_ids = model.encode(self.prompt)
        self.position = int(self.input_ids.shape[1]) - 1
        self.answer_ids = (
            _single_continuation_id(model, self.prompt, str(task.answer(1))),
            _single_continuation_id(model, self.prompt, str(task.answer(-1))),
        )
        with capture(model, list(range(model.n_layers))) as store:
            self.logits = model.forward_logits(self.input_ids)[0, -1].float().cpu()
        self.states = {
            layer: store.acts[layer][0][0, self.position].clone() for layer in range(model.n_layers)
        }

    def ids_for(self, sign: int) -> tuple[int, int]:
        """The (donor answer, other answer) token ids for one parity sign."""
        return (
            (self.answer_ids[0], self.answer_ids[1])
            if sign == 1
            else (
                self.answer_ids[1],
                self.answer_ids[0],
            )
        )

    def margin(self, sign: int) -> float:
        target, other = self.ids_for(sign)
        return float(self.logits[target] - self.logits[other])


def _lens(model: models.LoadedModel, state: Tensor) -> Tensor:
    """Read one residual through the frozen output head, logit-lens style."""
    inner = cast(Any, model.model)
    hidden = state.to(model.device, model.dtype).unsqueeze(0)
    return cast(Tensor, inner.lm_head(inner.model.norm(hidden)))[0].float().cpu()


@torch.no_grad()
def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {OUT}")
    preflight_check(MODEL, training=False)
    model = _load()
    started = time.time()
    try:
        clean = {
            (task.name, sign): Clean(model, task, sign) for task in ARITH_DEV for sign in (1, -1)
        }
        layers = list(range(model.n_layers))

        lens: dict[str, list[float]] = {}
        for (name, sign), record in clean.items():
            target, other = record.ids_for(sign)
            margins = []
            for layer in layers:
                read = _lens(model, record.states[layer])
                margins.append(float(read[target] - read[other]))
            lens[f"{name}:{sign:+d}"] = margins

        recovery: dict[int, list[float]] = {layer: [] for layer in layers}
        top1: dict[int, int] = dict.fromkeys(layers, 0)
        # How large the transplant actually is. A null is only informative if the
        # twin states differ enough for replacing one with the other to be an edit.
        relative_edit: dict[int, list[float]] = {layer: [] for layer in layers}
        for task in ARITH_DEV:
            for recipient_sign in (1, -1):
                donor_sign = -recipient_sign
                recipient = clean[(task.name, recipient_sign)]
                donor = clean[(task.name, donor_sign)]
                target, other = recipient.ids_for(donor_sign)
                for layer in layers:
                    delta = donor.states[layer] - recipient.states[layer]
                    relative_edit[layer].append(
                        float(delta.norm() / recipient.states[layer].norm())
                    )
                    with patch_residuals(
                        model,
                        layer,
                        (recipient.position,),
                        donor.states[layer].unsqueeze(0),
                    ):
                        patched = model.forward_logits(recipient.input_ids)[0, -1].float().cpu()
                    recovery[layer].append(
                        _normalized_recovery(
                            recipient.margin(donor_sign),
                            float(patched[target] - patched[other]),
                            donor.margin(donor_sign),
                        )
                    )
                    top1[layer] += int(int(patched.argmax()) == target)
            print(f"swept {task.name}", flush=True)

        mean_recovery = {
            str(layer): sum(values) / len(values) for layer, values in recovery.items()
        }
        best = max(layers, key=lambda layer: mean_recovery[str(layer)])
        summary = {
            "schema_version": 1,
            "kind": "post-hoc DEV localization diagnostic; not a gated result",
            "model": model.name,
            "model_revision": models.loaded_revision(model),
            "n_layers": model.n_layers,
            "site": "last pre-answer token",
            "bank": [task.name for task in ARITH_DEV],
            "n_transplants_per_layer": len(recovery[0]),
            "mean_relative_edit_size_by_layer": {
                str(layer): sum(values) / len(values) for layer, values in relative_edit.items()
            },
            "mean_normalized_recovery_by_layer": mean_recovery,
            "donor_answer_top1_by_layer": {str(layer): top1[layer] for layer in layers},
            "best_layer": best,
            "clean_answer_logit_lens_margin_by_layer": lens,
            "source_sha256": _sha256(Path(__file__)),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "elapsed_seconds": time.time() - started,
        }
        OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(mean_recovery))
        print(json.dumps(summary["donor_answer_top1_by_layer"]))
        print(f"best layer {best} at {mean_recovery[str(best)]:.3f}; wrote {OUT}")
    finally:
        model.free()


if __name__ == "__main__":
    main()
