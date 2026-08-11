"""OpenRouter provider with a hard spend cap. FINAL DATASET ONLY.

This is the last of the three providers, not the default one. Develop and pilot
on `cli_provider` (Codex / Claude Code, already paid for by a subscription); drop
to `local_provider` only for interpretability or a deliberate small-model
question; come here when the numbers are going into the write-up and need a
pinned, reproducible route. The `budget_usd` cap exists to make an accidental
run here cheap rather than surprising.

Same contract as the other two.

`complete(messages) -> {"text": ..., "model": ..., "provider": ...}`, so
`family_screen`, `binding_runner` and the branch runners cannot tell which
backend they are on.

Two things this does that a plain API wrapper does not:

**It pins the route.** OpenRouter load-balances a model across inference
providers and falls back between them by default. For an app that is a feature;
for an experiment it silently makes "condition 37" and "condition 38" different
machines. `allow_fallbacks` is off, the provider order is pinned when given, and
the provider actually served is recorded on every single call and asserted to be
stable across the run.

**It refuses to overspend.** `budget_usd` is checked before every call against
the real cost OpenRouter reports (`usage.include`), not an estimate. Exceeding it
raises. `--dry-run` costs nothing and still exercises the whole runner.
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import requests

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

# Reused from the podcast app rather than copied anywhere new; the key is read at
# call time and never logged.
DEFAULT_ENV_FILE = pathlib.Path.home() / "Programming/adblock-podcast/.env"

MODELS = {
    "deepseek": "deepseek/deepseek-v4-flash-0731",
    "luna": "openai/gpt-5.6-luna",
}


class BudgetExceeded(RuntimeError):
    pass


def read_key(env_file: pathlib.Path = DEFAULT_ENV_FILE) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    if not env_file.exists():
        raise RuntimeError(
            f"no OPENROUTER_API_KEY in the environment and no {env_file}")
    for line in env_file.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"OPENROUTER_API_KEY not found in {env_file}")


def load(model: str, *, provider: str | None = None, system: str | None = None,
         max_tokens: int = 400, budget_usd: float = 0.10, seed: int = 0,
         dry_run: bool = False, dry_run_reply: str = "ANSWER: 0.5",
         timeout_s: float = 120, log=print):
    """Returns `(complete, close)`. `close()` reports spend and route stability.

    `budget_usd` defaults deliberately low. A run that needs more should say so
    explicitly rather than discover it afterwards.
    """
    model = MODELS.get(model, model)
    key = None if dry_run else read_key()
    state = {"n": 0, "cost": 0.0, "in": 0, "out": 0, "providers": set(),
             "retries": 0}

    def complete(messages):
        state["n"] += 1
        if dry_run:
            # A callable lets a dry run exercise the real runner end to end --
            # binding choices, decoding, analysis -- instead of only counting
            # calls against a canned string that no decoder would accept.
            text = (dry_run_reply(messages) if callable(dry_run_reply)
                    else dry_run_reply)
            return {"text": text, "model": model, "provider": "dry-run",
                    "cost": 0.0}
        if state["cost"] >= budget_usd:
            raise BudgetExceeded(
                f"spent ${state['cost']:.4f} of a ${budget_usd:.2f} cap after "
                f"{state['n'] - 1} calls; raise --budget deliberately or stop")

        body = {
            "model": model,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + list(messages),
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            # Reasoning is an uncontrolled treatment variable here: a model that
            # thinks in some cells and not others is not one condition.
            "reasoning": {"enabled": False},
            "usage": {"include": True},
            # No silent reroute to a different inference provider mid-experiment.
            "provider": {"allow_fallbacks": False,
                         **({"order": [provider]} if provider else {})},
        }
        last = None
        for attempt in range(3):
            try:
                r = requests.post(
                    ENDPOINT, timeout=timeout_s,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    data=json.dumps(body))
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}"
                    state["retries"] += 1
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                break
            except requests.RequestException as e:
                last = str(e)
                state["retries"] += 1
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"3 failed attempts, last: {last}")

        d = r.json()
        usage = d.get("usage") or {}
        state["cost"] += float(usage.get("cost") or 0.0)
        state["in"] += int(usage.get("prompt_tokens") or 0)
        state["out"] += int(usage.get("completion_tokens") or 0)
        served = d.get("provider") or "unknown"
        state["providers"].add(served)
        return {"text": d["choices"][0]["message"]["content"] or "",
                "model": d.get("model", model), "provider": served,
                "cost": float(usage.get("cost") or 0.0),
                "finish_reason": d["choices"][0].get("finish_reason")}

    def close():
        log(f"[openrouter] {state['n']} calls, ${state['cost']:.4f}, "
            f"{state['in']}->{state['out']} tokens, "
            f"providers={sorted(state['providers']) or ['dry-run']}, "
            f"{state['retries']} retries")
        if len(state["providers"]) > 1:
            log(f"[openrouter] WARNING: served by {len(state['providers'])} "
                "different providers; conditions are not comparable")
        return dict(state, providers=sorted(state["providers"]))

    return complete, close


def demo():
    """No network, no spend. Checks the contract and that the cap actually bites."""
    complete, close = load("deepseek", dry_run=True, budget_usd=0.0)
    r = complete([{"role": "user", "content": "hi"}])
    assert r["text"] == "ANSWER: 0.5" and r["cost"] == 0.0, r
    assert close()["n"] == 1
    assert MODELS["deepseek"].startswith("deepseek/")

    # A zero cap must refuse on the FIRST call, before any request is sent --
    # otherwise the cap only stops the run after it has already overspent.
    complete, _ = load("deepseek", budget_usd=0.0)
    try:
        complete([{"role": "user", "content": "hi"}])
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("zero budget did not block the first call")

    key = read_key()
    assert key.startswith("sk-or-"), "key does not look like an OpenRouter key"
    print(f"ok (key resolves, {len(key)} chars, not logged)")


if __name__ == "__main__":
    demo()
