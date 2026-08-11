#!/usr/bin/env python3
"""Local model-free protocol preflight.

Runs syntax/unit tests and the shortcut simulator. It intentionally does not
pretend to validate GPU/model behavior.
"""
from __future__ import annotations
import importlib.util
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SRC=ROOT/"src"
sys.path.insert(0,str(SRC))

PY_FILES=list(SRC.glob("*.py")) + [
    ROOT/"run_dev.py", ROOT/"analyze_dev.py", ROOT/"evaluate_frontier.py",
    ROOT/"run_naive_semantic.py", ROOT/"analyze_naive_semantic.py", ROOT/"run_visible_dev.py",
    ROOT/"freeze_protocol.py", ROOT/"run_confirm.py", ROOT/"analyze_confirm.py",
    ROOT/"make_submission_figures.py",
    ROOT/"smoke_local.py", ROOT/"debug_source_effect.py", ROOT/"localize_source_layer.py",
    ROOT/"span_threshold.py", ROOT/"answer_site_probe.py",
]

def run_test_file(path: Path):
    spec=importlib.util.spec_from_file_location(path.stem,path)
    mod=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    for name in dir(mod):
        if name.startswith("test_"):
            getattr(mod,name)()

def main():
    for path in PY_FILES:
        py_compile.compile(str(path),doraise=True)
    tests=sorted((ROOT/"tests").glob("test_*.py"))
    for path in tests:
        run_test_file(path)
    sim=subprocess.run(
        [sys.executable,str(ROOT/"simulate_falsifiers.py")],
        cwd=ROOT,text=True,capture_output=True
    )
    if sim.returncode:
        raise RuntimeError(sim.stdout+"\n"+sim.stderr)
    print(f"syntax: {len(PY_FILES)} files OK")
    print(f"tests: {len(tests)} files OK")
    print("shortcut simulation: OK")
    print("GPU/model behavior: NOT TESTED by this preflight")

if __name__=="__main__":
    main()
