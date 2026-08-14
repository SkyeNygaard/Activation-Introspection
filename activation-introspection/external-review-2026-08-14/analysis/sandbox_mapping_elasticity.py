from pathlib import Path
import json,collections,statistics
import numpy as np
R=Path('/mnt/data/activation_introspection/extract3/results')
def jl(fn):return [json.loads(x) for x in (R/fn).read_text().splitlines() if x.strip()]
def mean(x):return float(np.mean(x)) if len(x) else float('nan')
out={}
for seed in range(3):
 rs=jl(f'remap_training_v2_seed{seed}_raw.jsonl')
 for arm in ['base','fixed','remap']:
  for cond in ['target','random']:
   z=[r for r in rs if r['arm']==arm and r['condition']==cond and float(r['strength'])==.5]
   g=collections.defaultdict(dict)
   for r in z:
    # raw Q-K = signed correct-label margin times sign(correct label Q=+)
    qk=float(r['signed_margin'])*(1 if r['correct_label']=='Q' else -1)
    key=(r['concept'],r['carrier_sha256'],r['order_key'],r['query_sign'])
    g[key][r['positive_label']]=qk
   pairs=[v for v in g.values() if set(v)=={'Q','K'}]
   a=np.array([v['Q'] for v in pairs]);b=np.array([v['K'] for v in pairs])
   # mapping Q vs K should reverse output for any fixed query sign: signs opposite
   flip=np.mean(np.sign(a)!=np.sign(b))
   # anti-correlation and antisymmetry normalized by total magnitude
   corr=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else float('nan')
   antisym=np.mean(np.abs(a+b)/(np.abs(a)+np.abs(b)+1e-9))
   delta=np.mean(np.abs(a-b))
   out[f'seed{seed}/{arm}/{cond}']={'n':len(pairs),'sign_flip_rate':float(flip),'qk_corr_across_mapping':corr,'normalized_antisymmetry_error':float(antisym),'mean_abs_mapping_effect':float(delta),'mean_abs_qk':float(np.mean((np.abs(a)+np.abs(b))/2))}
# aggregate seed means
agg={}
for arm in ['base','fixed','remap']:
 for cond in ['target','random']:
  vals=[out[f'seed{s}/{arm}/{cond}'] for s in range(3)]
  agg[f'{arm}/{cond}']={k:mean([v[k] for v in vals]) for k in vals[0] if k!='n'}
  agg[f'{arm}/{cond}']['n_total']=sum(v['n'] for v in vals)
P=Path('/mnt/data/activation_introspection/sandbox_mapping_elasticity_results.json');P.write_text(json.dumps({'per_seed':out,'aggregate':agg},indent=2));print(P)
for k,v in agg.items():print(k,v)
