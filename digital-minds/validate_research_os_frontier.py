#!/usr/bin/env python3
"""Model-free validation of the current M4 research frontier."""
from __future__ import annotations
import json, subprocess, sys, py_compile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FR=ROOT/"parallel_frontier"
WIN=ROOT/"winner_protocol"
SH=ROOT/"shared_behavioral"

checks=[]

def ok(name,condition):
    if not condition: raise AssertionError(name)
    checks.append(name)

# 1. All Python syntax.
for base in (FR,WIN,SH,ROOT/"m4_feasibility"):
    for p in base.rglob("*.py"):
        if "__pycache__" in p.parts: continue
        py_compile.compile(str(p),doraise=True)
ok("all_python_compiles",True)

# 2. Shared binding tasks.
r=subprocess.run([sys.executable,"test_shared_behavioral.py"],cwd=SH,
                 capture_output=True,text=True)
ok("shared_binding_tasks",r.returncode==0)

# 3. Goal-symbol structural balance includes adversarial cases.
r=subprocess.run([sys.executable,str(FR/"01_goal_relative_welfare"/"goal_protocol.py")],
                 capture_output=True,text=True)
ok("goal_counterbalance",r.returncode==0 and "all-even adversarial" in r.stdout)

# 4. Branch 04 design falsification.
# design_falsification.json is a static artifact, so also run the unit tests that
# exercise the analysis code -- otherwise these lines only check a file's contents.
B04=FR/"04_executed_budget_garp"
r=subprocess.run([sys.executable,"test_menu_rationality.py"],cwd=B04,
                 capture_output=True,text=True)
ok("skip_menu_rationality_unit_tests",r.returncode==0)
d=json.loads((B04/"design_falsification.json").read_text())
ok("skip_stable_utility_no_false_cycle",len(d["stable_log_utility_failures"])==0)
ok("skip_random_choice_rejection_power",d["random_choice_strict_cycle_rate"]>.95)
# Primary is the max-consistent fraction, not cycle existence: one slip by an
# otherwise rational agent trips the binary test but barely moves the fraction.
oc=d["one_slip_operating_characteristics"]
ok("skip_binary_test_is_brittle",oc["strict_cycle_rate"]["1"]>.4)
ok("skip_fraction_separates_one_slip_from_random",
   oc["mean_max_consistent_fraction"]["1"]
   > d["random_choice_mean_max_consistent_fraction_first_250"]+.2)

# 5. Branch 18 synthetic state pattern.
d=json.loads((FR/"18_preference_path_dependence"/"synthetic_ground_truth.json").read_text())
ok("path_dependence_recovers_satiation",d["dose3:full_history"]["mean"]<-.25)
ok("path_dependence_blank_near_zero",abs(d["dose3:blank_reset"]["mean"])<.06)

# 6. Branch 19 synthetic calibration.
d=json.loads((FR/"19_preference_self_knowledge"/"synthetic_ground_truth.json").read_text())
ok("preference_selfknowledge_overconfidence_fixture",
   d["naive_numeric"]["mean_signed_error"]>.10)
ok("preference_selfknowledge_structured_improves_fixture",
   d["structured"]["mae"]<d["naive_numeric"]["mae"])

# 7. Winner numeric scoring static audit.
render=(WIN/"src"/"rendering.py").read_text()
naive=(WIN/"src"/"naive_semantic.py").read_text()
dev=(WIN/"run_dev.py").read_text()
ok("winner_rating_space_prefix",'prefix = "Rating: "' in render)
ok("winner_bare_digit_options",'tuple(str(i) for i in range(10))' in naive
   and 'tuple(str(i) for i in range(10))' in dev)

# 8. Synthetic fixtures visibly quarantined.
ok("synthetic_fixture_warning",
   (WIN/"SYNTHETIC_FIXTURES_DO_NOT_TREAT_AS_RESULTS"/"README.md").exists())

print(json.dumps({"passed":len(checks),"checks":checks},indent=2))
