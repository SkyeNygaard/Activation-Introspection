#!/usr/bin/env python3
"""Additional kill-tests for sandbox_deep_pass_results.json."""
from __future__ import annotations
import json, itertools
from pathlib import Path
from statistics import mean
import numpy as np
import torch
from scipy import stats

ROOT=Path('/mnt/data/activation_introspection/work/repo')
RES=ROOT/'results'
OUT=Path('/mnt/data/activation_introspection')

def loadjl(name):
    return [json.loads(x) for x in (RES/name).read_text().splitlines() if x]

def demo_direction_convergence():
    profiles={}; concepts=None
    for seed in range(3):
        rows=loadjl(f'remap_training_v2_seed{seed}_raw.jsonl')
        concepts=sorted({r['concept'] for r in rows if r['condition']=='random'})
        for arm in ['base','fixed','remap']:
            profiles[(seed,arm)]=np.array([
                mean(bool(r['correct']) for r in rows if r['arm']==arm and r['condition']=='random' and float(r['strength'])==0.5 and r['concept']==c)
                for c in concepts])
    pooled={arm:np.stack([profiles[(s,arm)] for s in range(3)]).mean(0) for arm in ['base','fixed','remap']}
    out={'concepts':concepts,'pooled_accuracy':{a:dict(zip(concepts,map(float,p))) for a,p in pooled.items()}}
    for a,b in [('base','fixed'),('base','remap'),('fixed','remap')]:
        pr=stats.pearsonr(pooled[a],pooled[b]); sr=stats.spearmanr(pooled[a],pooled[b])
        out[f'{a}_vs_{b}']={'pearson_r':float(pr.statistic),'pearson_p':float(pr.pvalue),'spearman_rho':float(sr.statistic),'spearman_p':float(sr.pvalue)}
    return out

def dynamic_transport(path, strength=None):
    blob=torch.load(path,map_location='cpu',weights_only=False); A=blob['acts']; idx=blob['index']
    if strength is not None:
        keep=[i for i,m in enumerate(idx) if float(m.get('strength',1.0))==float(strength)]
        A=A[keep]; idx=[idx[i] for i in keep]
    depths=sorted({m['inject_layer'] for m in idx}); cars=sorted({m['carrier_id'] for m in idx}); cons=sorted({m['concept'] for m in idx})
    pos={(m['arm'],m['inject_layer'],m['carrier_id'],m['concept']):i for i,m in enumerate(idx)}
    out={'depths':depths,'final_layer':A.shape[1]-1,'arms':{}}
    for arm in ['target','random','shuffled']:
        pairrows=[]
        def mat(d,r):
            xs=[]
            for car in cars:
                x=torch.stack([A[pos[(arm,d,car,c)],r].float() for c in cons]); x=x-x.mean(0,keepdim=True); xs.append(x)
            x=torch.stack(xs).mean(0); return x/(x.norm(dim=1,keepdim=True)+1e-12)
        for d1,d2 in itertools.combinations(depths,2):
            X1=mat(d1,d2); X2=mat(d2,d2)
            start=float(torch.diag(X1@X2.T).mean())
            F1=mat(d1,A.shape[1]-1); F2=mat(d2,A.shape[1]-1)
            final=float(torch.diag(F1@F2.T).mean())
            pairrows.append({'early_inject':d1,'late_inject':d2,'cosine_at_late_injection_layer':start,'cosine_at_final':final,'delta':final-start})
        out['arms'][arm]={
            'mean_start_cosine':mean(r['cosine_at_late_injection_layer'] for r in pairrows),
            'mean_final_cosine':mean(r['cosine_at_final'] for r in pairrows),
            'mean_final_minus_start':mean(r['delta'] for r in pairrows),
            'fraction_pairs_increasing':mean(r['delta']>0 for r in pairrows),
            'pairs':pairrows,
        }
    return out

def heldout_generic_vs_floor():
    rows=loadjl('heldout_semantic_v1_raw.jsonl')
    cells={}
    for arm in sorted({r['arm'] for r in rows}):
        gs={}
        from collections import defaultdict
        g=defaultdict(list)
        for r in rows:
            if r['arm']==arm:g[(r['category_pair'],r['draw'],r['carrier_sha'],r['cell_base'])].append(r)
        cells[arm]={k:all(bool(x['model_correct']) for x in v) for k,v in g.items()}
    def cmp(a,b):
        ao=bo=both=neither=0
        for k in cells[a]:
            x,y=cells[a][k],cells[b][k]
            if x and y:both+=1
            elif x:ao+=1
            elif y:bo+=1
            else:neither+=1
        p=float(stats.binomtest(ao,ao+bo,.5).pvalue) if ao+bo else 1.0
        return {'a_only':ao,'b_only':bo,'both':both,'neither':neither,'exact_p':p}
    return {
        'semantic_vs_random':cmp('heldout_semantic','heldout_random'),
        'scrambled_vs_random':cmp('heldout_scrambled','heldout_random'),
        'semantic_vs_query_only':cmp('heldout_semantic','query_only'),
        'scrambled_vs_query_only':cmp('heldout_scrambled','query_only'),
        'warning':'Post-hoc comparisons; successes are exemplar/category-pair concentrated. They support weak generic hidden-state sensitivity, not semantic generalization.'
    }

def seen_vs_heldout():
    out={}
    from collections import defaultdict
    for seed in range(4):
        rows=loadjl(f'report_training_v3_seed{seed}_raw.jsonl')
        so={}
        for arm in ['trained','trained_seen_bank']:
            for cond in ['target','random','shuffled']:
                rs=[r for r in rows if r['arm']==arm and r['condition']==cond]
                slopes=[]; flips=[]
                for c in sorted({r['concept'] for r in rs}):
                    cr=[r for r in rs if r['concept']==c]
                    by=defaultdict(list)
                    for r in cr:by[r['carrier_sha256']].append(r)
                    css=[]
                    for g in by.values():
                        gp=next(x for x in g if x['sign']==1); gm=next(x for x in g if x['sign']==-1)
                        css.append((float(gp['signed_margin'])-(-float(gm['signed_margin'])))/2)
                        flips.append(gp['predicted_label']!=gm['predicted_label'])
                    slopes.append(mean(css))
                so[f'{arm}/{cond}']={'mean_abs_slope':mean(abs(x) for x in slopes),'flip_rate':mean(flips)}
        out[str(seed)]=so
    return out

def main():
    d=json.load(open(OUT/'sandbox_deep_pass_results.json'))
    extra={
      'demonstrated_random_direction_difficulty_convergence':demo_direction_convergence(),
      'dynamic_transport_kill_test':{
        'qwen05b_test_repaired':dynamic_transport(RES/'retained_test_qwen05b_v2_raw.acts.pt'),
        'qwen15b_dev_repaired':dynamic_transport(RES/'retained_dev_qwen15b_raw.acts.pt',1.0),
        'qwen3b_dev_repaired':dynamic_transport(RES/'retained_dev_qwen3b_raw.acts.pt',1.0),
      },
      'heldout_generic_vs_floor_posthoc':heldout_generic_vs_floor(),
      'zero_demo_seen_vs_heldout_response':seen_vs_heldout(),
    }
    d['extra_kill_tests']=extra
    (OUT/'sandbox_deep_pass_results.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
    (OUT/'sandbox_extra_pass_results.json').write_text(json.dumps(extra,indent=2,sort_keys=True)+'\n')
    print('fixed/remap direction correlation',extra['demonstrated_random_direction_difficulty_convergence']['fixed_vs_remap'])
    print('dynamic target', {m:v['arms']['target'] for m,v in extra['dynamic_transport_kill_test'].items()})
    print('heldout generic',extra['heldout_generic_vs_floor_posthoc'])
if __name__=='__main__':main()
