# DEV → Freeze → Confirmation Governance

## Development

You may inspect DEV outputs while selecting:
- model;
- layer;
- steering factor.

DEV carrier prompts are not confirmation evidence.

## Freeze

Once a cell passes the joint DEV gates:

```bash
python freeze_protocol.py \
  --model <MODEL> \
  --vector-file <VECTOR_PATH> \
  --layer <LAYER> \
  --factor <FACTOR> \
  --selected-dev-summary <DEV_SUMMARY_JSON> \
  --out protocol_confirm.json
```

This resolves and stores immutable Hugging Face model/vector revisions, the exact
12 confirmation carriers, persona text, success rules, and SHA-256 hashes of all
code used by confirmation.

Commit `protocol_confirm.json` **before** running confirmation.

## Confirmation

```bash
python run_confirm.py \
  --protocol protocol_confirm.json \
  --out results/confirm_raw.jsonl

python analyze_confirm.py \
  --protocol protocol_confirm.json \
  --raw results/confirm_raw.jsonl \
  --out results/confirm_summary.json
```

The runner refuses overwrite and refuses source/prompt drift.

## After confirmation starts

Do not change:
- model/vector;
- layer/factor;
- carrier list;
- persona prompts;
- readout wording;
- arms;
- target-query primary contrast;
- success threshold.

Bugs that invalidate the execution can justify a repaired prospective
confirmation, but:
1. retain the failed/inspected artifact;
2. document the defect;
3. freeze the repair before rerun;
4. use fresh confirmation carriers if the original outcomes were exposed enough
   to guide substantive design choices.

That is exactly the standard that preserved credibility in the prior codebook
study after its inspected precursor required repair.


## Modal confirmation

After `protocol_confirm.json` is committed:

```bash
modal run modal_confirm.py --protocol protocol_confirm.json
```

The Modal wrapper accepts **no model/layer/factor overrides**. Those come only
from the frozen protocol. It writes the raw rows, manifest, summary, and run log
back locally.

The scientific runner records Python, PyTorch, Transformers, and
huggingface_hub versions in the manifest.
