"""The default way to run an experiment here: script the Codex / Claude Code CLIs.

Same `complete(messages)` contract as `local_provider` and
`openrouter_provider`, so a runner cannot tell the difference.

**Reach for this first.** Both draw on an existing subscription rather than
per-token API spend, so there is no cost decision to make per call and no reason
to under-power a pilot. The order across the three providers is:

  1. this module -- develop, debug and pilot everything here;
  2. `local_provider` -- only when there is a reason: white-box or
     interpretability work that needs activations, or a question genuinely about
     small models;
  3. `openrouter_provider` -- the final dataset for the write-up, and nothing
     else.

## Why this is not also step 3

**They are agent harnesses, not model endpoints.** Each wraps the model in its
own system instructions, tool affordances and agent context. For a coding
benchmark that is the point; for an experiment asking *what does this model
prefer*, the harness is exactly the kind of thing that could be producing the
preference. The honest subject line for a Codex run is "GPT-5.6 Luna in the
Codex agent environment", never "GPT-5.6 Luna" -- so `provider` is reported as
`codex-harness:<model>` and there is no way to get a bare model name out of this
module.

**Multi-turn fidelity is lost.** Both CLIs take one prompt, so a message list is
flattened into role-marked text. The model sees a transcript rather than
occupying it. Branch 18/20 treatments depend on the model having *been through*
the work, and a flattened transcript is a weaker version of that.

**Sampling is not controlled.** Neither exposes temperature or a seed here, so
runs are not reproducible the way a pinned OpenRouter call is.

So: everything up to and including "is there a real effect here" belongs on this
path. Only the numbers that get printed in the report need a pinned API route.
"""
from __future__ import annotations

import json
import subprocess

HARNESS_WARNING = (
    "CLI harness output is pilot-only: the model is wrapped in agent system "
    "instructions and the message list is flattened to one prompt"
)


def flatten(messages, system: str | None = None) -> str:
    """Role-marked transcript. The last user turn is the live question."""
    parts = []
    if system:
        parts.append(f"[instructions]\n{system}")
    for m in messages:
        parts.append(f"[{m['role']}]\n{m['content']}")
    return "\n\n".join(parts)


def _run(cmd, stdin_text: str, timeout_s: float) -> str:
    p = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True,
                       timeout=timeout_s)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {p.returncode}: {p.stderr[-400:]}")
    return p.stdout


def _codex_final(stdout: str) -> str:
    """Last agent message from `codex exec --json` JSONL.

    The payload is `{"type": "item.completed", "item": {"type":
    "agent_message", "text": ...}}`. The event type alone is not enough -- a
    tool call is also an `item.completed` -- so the inner item type is what
    selects the message. Older builds nest the same thing under `msg`.
    """
    text = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        for payload in (ev.get("item"), ev.get("msg"), ev):
            if not isinstance(payload, dict):
                continue
            if payload.get("type") not in ("agent_message", "assistant_message"):
                continue
            for key in ("text", "message", "content"):
                v = payload.get(key)
                if isinstance(v, str) and v.strip():
                    text = v
    if not text:
        raise RuntimeError(f"no agent_message in codex output: {stdout[-300:]!r}")
    return text


def _claude_final(stdout: str) -> str:
    """`claude -p --output-format json` returns one object with `result`.

    An error is reported in that same `result` field with `is_error: true`, so
    "Not logged in - Please run /login" arrives looking exactly like a model
    reply. Raise on it: a harness failure that reaches the analysis as a datum is
    worse than a crash.
    """
    try:
        d = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip()
    if isinstance(d, dict):
        if d.get("is_error"):
            raise RuntimeError(f"claude harness error: {d.get('result')!r}")
        for key in ("result", "text", "content"):
            v = d.get(key)
            if isinstance(v, str):
                return v
    return stdout.strip()


def load(harness: str, *, model: str | None = None, system: str | None = None,
         timeout_s: float = 180, log=print):
    """`harness` is 'codex' or 'claude'. Returns `(complete, close)`."""
    if harness not in ("codex", "claude"):
        raise ValueError(f"unknown harness {harness!r}")
    log(f"[cli_provider] {HARNESS_WARNING}")
    state = {"n": 0, "failures": 0}

    def complete(messages):
        state["n"] += 1
        prompt = flatten(messages, system)
        if harness == "codex":
            # Isolation flags are not optional for a preference experiment. Without
            # them Codex reads ~/.codex/AGENTS.md, the project doc and the user's
            # rules into every trial, so Skye's own standing instructions would sit
            # inside the context that the model's task choices are measured from.
            # `read-only` also stops a trial from touching the filesystem while
            # "doing the work".
            cmd = ["codex", "exec", "--json", "--ephemeral",
                   "--skip-git-repo-check", "--sandbox", "read-only",
                   "--ignore-user-config", "--ignore-rules",
                   "-c", "project_doc_max_bytes=0",
                   "--disable", "plugins", "--disable", "apps",
                   "--disable", "hooks", "--disable", "memories",
                   "--disable", "skill_search", "--disable", "multi_agent", "-"]
            if model:
                cmd[-1:] = ["-m", model, "-"]
            text = _codex_final(_run(cmd, prompt, timeout_s))
        else:
            # No --bare: it drops hooks and plugins, which would be welcome, but
            # it also never reads OAuth or the keychain, so every call comes back
            # "Not logged in". The harness scaffolding stays.
            cmd = ["claude", "-p", "--output-format", "json"]
            if model:
                cmd += ["--model", model]
            text = _claude_final(_run(cmd, prompt, timeout_s))
        if not text.strip():
            state["failures"] += 1
        return {"text": text, "model": model or "harness-default",
                "provider": f"{harness}-harness:{model or 'default'}",
                "pilot_only": True}

    def close():
        log(f"[cli_provider] {state['n']} calls via {harness}, "
            f"{state['failures']} empty. {HARNESS_WARNING}")
        return dict(state, harness=harness, pilot_only=True)

    return complete, close


def demo():
    """Contract check without invoking either CLI."""
    msgs = [{"role": "user", "content": "pick one"},
            {"role": "assistant", "content": "Q"},
            {"role": "user", "content": "now do it"}]
    f = flatten(msgs, system="be brief")
    assert f.startswith("[instructions]\nbe brief"), f
    assert f.count("[user]") == 2 and "[assistant]" in f

    assert _claude_final('{"result": "ANSWER: 0.4"}') == "ANSWER: 0.4"
    assert _claude_final("not json") == "not json"
    # A harness error arrives in the same field a real answer would.
    try:
        _claude_final('{"is_error": true, "result": "Not logged in"}')
    except RuntimeError:
        pass
    else:
        raise AssertionError("harness error was returned as a model reply")
    jsonl = ('{"type": "thread.started", "thread_id": "x"}\n'
             '{"type": "item.completed", "item": {"type": "agent_message", '
             '"text": "first"}}\n'
             'noise\n'
             '{"type": "item.completed", "item": {"type": "command_execution", '
             '"text": "ls"}}\n'
             '{"type": "item.completed", "item": {"type": "agent_message", '
             '"text": "ANSWER: Q"}}\n')
    # The LAST agent message is the answer, and a tool call is also an
    # item.completed -- selecting on the event type alone returns "ls".
    assert _codex_final(jsonl) == "ANSWER: Q"
    try:
        _codex_final('{"type": "turn.completed"}')
    except RuntimeError:
        pass
    else:
        raise AssertionError("codex output with no agent message was accepted")

    for bad in ("gpt", "anthropic"):
        try:
            load(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted harness {bad!r}")
    print("ok")


if __name__ == "__main__":
    demo()
