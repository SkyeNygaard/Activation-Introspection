from pathlib import Path
import json,re,statistics,math
from collections import defaultdict
import numpy as np
R=Path('/mnt/data/activation_introspection/extract3/results')

def jl(fn):return [json.loads(x) for x in (R/fn).read_text().splitlines() if x.strip()]
def avg(x):return sum(x)/len(x) if x else float('nan')
# zero-demo Q-K raw margins per seed/arm/condition/concept/carrier
native=defaultdict(list)
for seed in range(4):
 rows=jl(f'report_training_v3_seed{seed}_raw.jsonl')
 for r in rows:
  if r['condition']=='clean':continue
  # signed margin is margin of expected label; Q-K raw reverses when expected K
  raw=float(r['signed_margin'])*(1 if r['expected_label']=='Q' else -1)
  native[(seed,r['arm'],r['condition'],r['concept'],r['carrier_sha256'],r['sign'])].append(raw)
# slope: (QK(+)-QK(-))/2
slopes={}
for seed in range(4):
 for arm in ['base','trained']:
  for cond in ['target','random','shuffled']:
   concepts=sorted({k[3] for k in native if k[0]==seed and k[1]==arm and k[2]==cond})
   carriers=sorted({k[4] for k in native if k[0]==seed and k[1]==arm and k[2]==cond})
   for c in concepts:
    for car in carriers:
     try:
      p=avg(native[(seed,arm,cond,c,car,1)]);m=avg(native[(seed,arm,cond,c,car,-1)])
      slopes[(seed,arm,cond,c,car)]=(p-m)/2
     except: pass
# mean trained slope across report LoRA seeds
mean_tr={}
for cond in ['target','random']:
 for c in sorted({k[3] for k in slopes if k[1]=='trained' and k[2]==cond}):
  for car in sorted({k[4] for k in slopes if k[1]=='trained' and k[2]==cond}):
   vals=[slopes[(s,'trained',cond,c,car)] for s in range(4) if (s,'trained',cond,c,car) in slopes]
   if vals:mean_tr[(cond,c,car)]=avg(vals)
# remap rows all 3 seeds
out={}
for seed in range(3):
 rows=jl(f'remap_training_v2_seed{seed}_raw.jsonl')
 for arm in ['base','fixed','remap']:
  for cond in ['target','random']:
   z=[r for r in rows if r['arm']==arm and r['condition']==cond and float(r['strength'])==.5]
   # alignment source: base exact seed0 zero-demo base (base identical all seeds); trained mean for fixed/remap
   zz=[]
   for r in z:
    if arm=='base': sl=slopes.get((0,'base',cond,r['concept'],r['carrier_sha256']))
    else: sl=mean_tr.get((cond,r['concept'],r['carrier_sha256']))
    if sl is None or sl==0:continue
    native_pos='Q' if sl>0 else 'K'
    aligned=(r['positive_label']==native_pos)
    rr=dict(r);rr['aligned_to_native']=aligned;rr['native_slope']=sl;zz.append(rr)
   for a in [True,False]:
    q=[r for r in zz if r['aligned_to_native']==a]
    key=f'seed{seed}/{arm}/{cond}/'+('aligned' if a else 'anti')
    out[key]={'n':len(q),'row_acc':avg([r['correct'] for r in q]),'mean_signed_margin':avg([r['signed_margin'] for r in q])}
   # twin pairs within mapping alignment: same cell base strip q sign
   g=defaultdict(list)
   for r in zz:
    base=re.sub(r'q[+-]1$','q',r['cell_id']);g[(r['concept'],r['carrier_sha256'],base,r['positive_label'],r['aligned_to_native'])].append(r)
   for a in [True,False]:
    pairs=[v for k,v in g.items() if k[-1]==a and len(v)==2]
    out[f'seed{seed}/{arm}/{cond}/'+('aligned' if a else 'anti')]['twin_acc']=avg([all(x['correct'] for x in v) for v in pairs])
# aggregate across seeds weighted equally rows
agg={}
for arm in ['base','fixed','remap']:
 for cond in ['target','random']:
  for a in ['aligned','anti']:
   vals=[out[f'seed{s}/{arm}/{cond}/{a}'] for s in range(3)]
   agg[f'{arm}/{cond}/{a}']={m:avg([v[m] for v in vals]) for m in ['row_acc','twin_acc','mean_signed_margin']}
   agg[f'{arm}/{cond}/{a}']['n_total']=sum(v['n'] for v in vals)
out2={'per_seed':out,'aggregate':agg,'native_slopes_base_random':{f'{c}/{car[:8]}':sl for (s,a,co,c,car),sl in slopes.items() if s==0 and a=='base' and co=='random'},'native_slopes_trained_random_mean':{f'{c}/{car[:8]}':sl for (co,c,car),sl in mean_tr.items() if co=='random'}}
P=Path('/mnt/data/activation_introspection/sandbox_override_results.json');P.write_text(json.dumps(out2,indent=2));print(P)
for k,v in agg.items():print(k,v)
