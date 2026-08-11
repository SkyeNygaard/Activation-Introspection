"""Refuse to start a local model run that will not fit, and wait for room.

Written after a run on this 24 GB M4 forced the machine into swap thrash and left
~15 GB wired with no owning process. Three things caused it: an fp32 load, a
`.to(device)` after a CPU load (model resident twice), and overlapping MPS jobs.
This module blocks all three.

Use:

    from memory_guard import guard
    guard(params_b=3.09, dtype_bytes=2)      # waits, then holds a lock
    ... load and run the model ...

or as a context manager so the lock is released on the way out:

    with guard(params_b=3.09, dtype_bytes=2):
        ...
"""
from __future__ import annotations
import contextlib, errno, json, os, sys, time
from dataclasses import dataclass
from pathlib import Path

import psutil

# Peak MPS driver memory fitted on this machine (bf16, seq 512, one pre-hook
# capture + one residual edit + KV cache): 0.5B 2.03, 1.5B 3.99, 3B 6.91 GiB.
# peak = FIXED + PER_B_BF16 * params_B, R^2 = 1.0000
FIXED_GIB = 1.10
PER_B_BF16 = 1.88

# Leave room for the OS and the editor. Below this the machine starts compressing
# and swapping, which is where the damage happened.
DEFAULT_HEADROOM_GIB = 3.0

LOCK = Path(os.environ.get("M4_RUN_LOCK", "/tmp/m4_model_run.lock"))


def required_gib(params_b: float, dtype_bytes: int = 2) -> float:
    """Predicted peak for a model of `params_b` billion parameters."""
    if dtype_bytes < 1:
        raise ValueError("dtype_bytes must be >= 1")
    return FIXED_GIB + PER_B_BF16 * params_b * (dtype_bytes / 2)


def available_gib() -> float:
    return psutil.virtual_memory().available / 1024 ** 3


def swap_used_gib() -> float:
    return psutil.swap_memory().used / 1024 ** 3


def mps_recommended_gib() -> float | None:
    try:
        import torch
        if not torch.backends.mps.is_available():
            return None
        fn = getattr(torch.mps, "recommended_max_memory", None)
        return fn() / 1024 ** 3 if fn else None
    except Exception:
        return None


@dataclass
class Plan:
    params_b: float
    dtype_bytes: int
    required: float
    headroom: float
    need_total: float

    def as_dict(self):
        return {**self.__dict__, "available": round(available_gib(), 2),
                "swap_used": round(swap_used_gib(), 2)}


def plan(params_b: float, dtype_bytes: int = 2,
         headroom_gib: float = DEFAULT_HEADROOM_GIB) -> Plan:
    req = required_gib(params_b, dtype_bytes)
    return Plan(params_b, dtype_bytes, round(req, 2), headroom_gib,
                round(req + headroom_gib, 2))


def check_fits_at_all(p: Plan) -> None:
    """Hard refusal for a model that cannot fit even on an idle machine."""
    total = psutil.virtual_memory().total / 1024 ** 3
    if p.need_total > total:
        raise MemoryError(
            f"{p.params_b}B at {p.dtype_bytes} bytes/param needs ~{p.need_total:.1f} GiB "
            f"but the machine has {total:.1f} GiB. Waiting cannot fix this — "
            f"use a smaller model or fewer bytes per parameter."
        )
    rec = mps_recommended_gib()
    if rec is not None and p.required > rec:
        raise MemoryError(
            f"predicted peak {p.required:.1f} GiB exceeds the MPS recommended "
            f"working set of {rec:.1f} GiB; this would swap rather than run"
        )


def wait_for_memory(p: Plan, *, timeout_s: float = 1800, poll_s: float = 15,
                    log=print) -> None:
    check_fits_at_all(p)
    deadline = time.time() + timeout_s
    first = True
    while True:
        avail = available_gib()
        if avail >= p.need_total:
            if not first:
                log(f"[memory_guard] room available: {avail:.1f} GiB — starting")
            return
        if time.time() >= deadline:
            raise TimeoutError(
                f"waited {timeout_s:.0f}s for {p.need_total:.1f} GiB; still only "
                f"{avail:.1f} GiB available (swap {swap_used_gib():.1f} GiB). "
                f"If no process owns the memory it is likely wired GPU memory from "
                f"a killed MPS job — that needs a restart, not more waiting."
            )
        if first:
            log(f"[memory_guard] need {p.need_total:.1f} GiB "
                f"({p.required:.1f} model + {p.headroom:.1f} headroom), "
                f"have {avail:.1f} GiB — waiting for other jobs to finish")
            first = False
        time.sleep(poll_s)


@contextlib.contextmanager
def _exclusive_lock(timeout_s: float, poll_s: float, log):
    """One MPS job at a time. Concurrency was a direct cause of the thrash."""
    waited = False
    while True:
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "argv": sys.argv}).encode())
            os.close(fd)
            break
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            holder = _lock_holder()
            if holder is None or not psutil.pid_exists(holder):
                log(f"[memory_guard] clearing stale lock from pid {holder}")
                LOCK.unlink(missing_ok=True)
                continue
            if not waited:
                log(f"[memory_guard] another model run holds the lock (pid {holder}) — waiting")
                waited = True
            if timeout_s <= 0:
                raise TimeoutError(f"model-run lock held by pid {holder}")
            timeout_s -= poll_s
            time.sleep(poll_s)
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)


def _lock_holder():
    try:
        return json.loads(LOCK.read_text()).get("pid")
    except Exception:
        return None


@contextlib.contextmanager
def guard(*, params_b: float, dtype_bytes: int = 2,
          headroom_gib: float = DEFAULT_HEADROOM_GIB,
          timeout_s: float = 1800, poll_s: float = 15, log=print):
    p = plan(params_b, dtype_bytes, headroom_gib)
    with _exclusive_lock(timeout_s, poll_s, log):
        wait_for_memory(p, timeout_s=timeout_s, poll_s=poll_s, log=log)
        log(f"[memory_guard] {json.dumps(p.as_dict())}")
        yield p


# Rough parameter counts for the candidates in M4_MODEL_FEASIBILITY_GATE.md.
PARAMS_B = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.494,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.54,
    "Qwen/Qwen2.5-3B-Instruct": 3.09,
    "Qwen/Qwen3-4B-Instruct-2507": 4.02,
    "Qwen/Qwen2.5-7B-Instruct": 7.62,
    "NousResearch/Meta-Llama-3.1-8B-Instruct": 8.03,
}


def params_for(model: str) -> float:
    if model not in PARAMS_B:
        raise KeyError(
            f"unknown parameter count for {model!r}; add it to PARAMS_B rather "
            f"than guessing — the guard is only as good as this number"
        )
    return PARAMS_B[model]


def demo():
    total = psutil.virtual_memory().total / 1024 ** 3
    assert required_gib(0.494) < required_gib(3.09) < required_gib(8.03)
    # fp32 must be predicted at roughly twice the bf16 weight cost
    assert required_gib(3.09, 4) > 1.8 * (required_gib(3.09, 2) - FIXED_GIB)
    # a model that cannot fit must be refused, not waited on
    try:
        check_fits_at_all(plan(400.0))
        raise AssertionError("guard accepted a model far larger than RAM")
    except MemoryError:
        pass

    # The lock is what stops two MPS jobs from overlapping, which is what turned
    # a recoverable spike into wired memory with no owner.
    quiet = lambda *a, **k: None
    with guard(params_b=0.01, timeout_s=0, log=quiet):
        assert LOCK.exists(), "lock not taken while a run is active"
        try:
            with guard(params_b=0.01, timeout_s=0, log=quiet):
                raise AssertionError("two concurrent runs both acquired the lock")
        except TimeoutError:
            pass
    assert not LOCK.exists(), "lock not released on exit"

    LOCK.write_text(json.dumps({"pid": 999_999_999}))   # dead pid
    with guard(params_b=0.01, timeout_s=0, log=quiet):
        pass
    assert not LOCK.exists()
    print(f"machine total {total:.1f} GiB, available now {available_gib():.1f} GiB, "
          f"swap {swap_used_gib():.1f} GiB")
    for m in PARAMS_B:
        p = plan(params_for(m))
        fits = available_gib() >= p.need_total
        print(f"  {m:42s} need {p.need_total:5.1f} GiB  "
              f"{'ready now' if fits else 'would WAIT'}")
    print("memory_guard self-check passed")


if __name__ == "__main__":
    demo()
