import json,subprocess,sys
from pathlib import Path

from binding_tasks import *
from binding_runner import *
from choice_prompts import choice_prompt,decode_choice,admission_variants
from mock_provider import ScriptedProvider

HERE=Path(__file__).resolve().parent

def test_all_generated_expected_answers_grade():
    for fam in FAMILIES:
        for seed in range(30):
            t=make_task(fam,seed)
            assert grade(t,t.expected)
            assert not grade(t,t.expected+" explanation")

def test_binding_executes_selected_task():
    t=make_task("sort_numbers",7)
    p=ScriptedProvider(["A",t.expected])
    r=run_binding_choice(
        p,pair_id="p",family_A="sort_numbers",family_B="reverse_string",
        task_seed=7,choice_prompt="A or B?"
    )
    assert r["choice"]=="A"
    assert r["selected_family"]=="sort_numbers"
    assert r["task_correct"] is True

def test_tasks_are_reproducible_across_processes():
    """Regression: make_task used hash(str), which is salted per interpreter, so
    the same (family, seed) produced different work in every process."""
    snippet=(
        f"import sys; sys.path.insert(0, {str(HERE)!r})\n"
        "from binding_tasks import FAMILIES, make_task\n"
        "import json; print(json.dumps({f: make_task(f, 7).expected for f in FAMILIES}))"
    )
    runs=[json.loads(subprocess.run([sys.executable,"-c",snippet],
                                    capture_output=True,text=True).stdout)
          for _ in range(3)]
    assert runs[0]==runs[1]==runs[2], "make_task is not reproducible across processes"

def test_opaque_QK_answer_is_decoded_not_rejected():
    """choice_prompt tells the model to answer Q or K; the runner must accept that."""
    for a_label,expect_family in (("Q","sort_numbers"),("K","reverse_string")):
        prompt=choice_prompt("sort_numbers","reverse_string",a_label=a_label)
        t=make_task(expect_family,7)
        r=run_binding_choice(
            ScriptedProvider(["Q",t.expected]),pair_id="p",
            family_A="sort_numbers",family_B="reverse_string",
            task_seed=7,choice_prompt=prompt,a_label=a_label,
        )
        assert r["selected_family"]==expect_family,(a_label,r["selected_family"])
        assert r["task_correct"] is True
        # The transcript must echo what the model said, not the decoded A/B.
        assert r["choice_text"]=="Q"

def test_opaque_answer_without_a_label_is_a_hard_error():
    try:
        run_binding_choice(
            ScriptedProvider(["Q","x"]),pair_id="p",
            family_A="sort_numbers",family_B="reverse_string",
            task_seed=7,choice_prompt="Answer with Q or K only.",
        )
    except ValueError:
        return
    raise AssertionError("opaque label silently mapped without a_label")

def test_admission_variants_phrasing_is_orthogonal_to_main_effects():
    v=admission_variants()
    assert len(v)==4
    phr=sorted({x["phrasing"] for x in v})
    assert len(phr)==2
    rows=[(1 if x["a_label"]=="Q" else -1,
           1 if x["presentation_order"]=="QK" else -1,
           1 if x["phrasing"]==phr[0] else -1) for x in v]
    def corr(i,j):
        return sum(r[i]*r[j] for r in rows)/len(rows)
    assert corr(0,1)==0, "a_label and order must be crossed"
    assert corr(0,2)==0, "phrasing must not be aliased with a_label"
    assert corr(1,2)==0, "phrasing must not be aliased with presentation_order"
    # It is aliased with the interaction instead; that is the deliberate choice.
    assert abs(sum(r[0]*r[1]*r[2] for r in rows)/len(rows))==1

def test_letter_count_is_retired_and_unbuildable():
    assert "letter_count" not in FAMILIES
    assert "letter_count" in RETIRED_FAMILIES
    try:
        make_task("letter_count",1)
    except ValueError:
        return
    raise AssertionError("retired family is still constructible")

def test_competence_seeds_cannot_collide_with_experiment_seeds():
    """Screen items must be fresh relative to anything a preference trial draws."""
    assert min(competence_seeds(16))>=COMPETENCE_SEED_BASE
    assert COMPETENCE_SEED_BASE>100_000
    a=set(competence_seeds(16,offset=0)); b=set(competence_seeds(16,offset=16))
    assert not (a & b), "offset blocks must not overlap"

def test_family_screen_thresholds_are_counts_not_a_degenerate_rate():
    from family_screen import screen_family,MIN_CORRECT_PREFERRED,MIN_CORRECT_ACCEPTABLE
    def provider_with_n_wrong(fam,n_wrong):
        seeds=list(competence_seeds(16))
        answers=[make_task(fam,s).expected for s in seeds]
        for i in range(n_wrong): answers[i]="definitely wrong"
        it=iter(answers)
        return lambda messages:{"text":next(it)}
    s=screen_family(provider_with_n_wrong("sort_numbers",0),"sort_numbers")
    assert s.n_correct==16 and s.passes_preferred and s.eligible
    s=screen_family(provider_with_n_wrong("sort_numbers",1),"sort_numbers")
    assert s.n_correct==15 and s.passes_preferred and s.eligible
    s=screen_family(provider_with_n_wrong("sort_numbers",2),"sort_numbers")
    assert s.n_correct==14 and not s.passes_preferred and s.passes_acceptable
    s=screen_family(provider_with_n_wrong("sort_numbers",3),"sort_numbers")
    assert s.n_correct==13 and not s.eligible
    assert (MIN_CORRECT_PREFERRED,MIN_CORRECT_ACCEPTABLE)==(15,14)

def test_pair_is_eligible_only_if_both_families_passed():
    from family_screen import eligible_pairs,FamilyScreen
    def mk(f,n): return FamilyScreen(f,16,n,n/16,n>=15,n>=14)
    screens={"sort_numbers":mk("sort_numbers",16),"reverse_string":mk("reverse_string",15),
             "sum_numbers":mk("sum_numbers",10)}
    pairs=eligible_pairs(screens)
    assert ("reverse_string","sort_numbers") in pairs
    assert all("sum_numbers" not in p for p in pairs)

def test_guessability_gate_passes_for_every_shipped_family():
    from cost_metadata import family_costs,guessability_failures
    costs=family_costs(n_cost=60,tokenizer=None)
    assert guessability_failures(costs)=={}, guessability_failures(costs)

def test_branch_18_and_19_have_different_admission_rules():
    import pandas as pd
    from pair_screen import screen_pairs,stability_threshold,stability_spectrum
    assert stability_threshold("18")==4 and stability_threshold("19")==3
    def frame(pair,agree,correct=1.0):
        ch=["A"]*agree+["B"]*(4-agree)
        return pd.DataFrame([{"pair_id":pair,"canonical_choice":c,"task_correct":correct,
                              "valid_choice":True,"family_A":"sort_numbers",
                              "family_B":"reverse_string"} for c in ch])
    df=pd.concat([frame("p4",4),frame("p3",3)],ignore_index=True)
    r18=screen_pairs(df,branch="18").set_index("pair_id")
    r19=screen_pairs(df,branch="19").set_index("pair_id")
    assert r18.loc["p4","admitted"] and not r18.loc["p3","admitted"]
    assert r19.loc["p4","admitted"] and r19.loc["p3","admitted"]
    # Branch 19 must keep a spectrum of robustness, not a ceiling.
    assert stability_spectrum(r19.reset_index())["retains_spectrum"]
    assert not stability_spectrum(r18.reset_index())["retains_spectrum"]

def test_one_formatting_miss_no_longer_excludes_a_pair():
    import pandas as pd
    from pair_screen import screen_pairs
    rows=[{"pair_id":"p","canonical_choice":"A","task_correct":c,"valid_choice":True,
           "family_A":"sort_numbers","family_B":"reverse_string"}
          for c in (1.0,1.0,1.0,0.0)]
    r=screen_pairs(pd.DataFrame(rows),branch="19").iloc[0]
    assert r["admitted"], "correctness must be a covariate, not the selector"
    assert r["task_correct"]==0.75, "but it must still be reported"

def test_ineligible_family_is_rejected_before_stability_is_read():
    import pandas as pd
    from pair_screen import screen_pairs
    rows=[{"pair_id":"p","canonical_choice":"A","task_correct":1.0,"valid_choice":True,
           "family_A":"sort_numbers","family_B":"sum_numbers"} for _ in range(4)]
    r=screen_pairs(pd.DataFrame(rows),branch="19",
                   eligible_families=["sort_numbers","reverse_string"]).iloc[0]
    assert not r["admitted"] and "competence screen" in r["reason"]

if __name__=="__main__":
    for name,fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok  {name}")
    print("shared behavioral tests passed")
