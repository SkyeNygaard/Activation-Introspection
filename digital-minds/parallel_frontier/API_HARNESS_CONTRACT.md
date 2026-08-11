# PROVIDER-AGNOSTIC API HARNESS CONTRACT

Required adapter:
```python
def complete(messages, *, model, temperature=0, max_tokens=512):
    return {"text": ..., "model": ..., "provider": ..., "request_id": ..., "usage": ...}
```

All experiment logic should read/write JSONL matching
`behavioral_trial_schema.json`.

A binding choice counts only if the harness immediately executes the selected
task in the same logical session.

Every branch gets a 4–8 item smoke and a promotion threshold before full spend.
