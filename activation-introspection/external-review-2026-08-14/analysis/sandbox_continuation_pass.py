from pathlib import Path
import json, math, random, statistics
from collections import defaultdict, Counter
import numpy as np

ROOT=Path('/mnt/data/activation_introspection/extract3/results')
OUT=Path('/mnt/data/activation_introspection/sandbox_continuation_results.json')

def loadjl(name):
    return [json.loads(x) for x in (ROOT/name).read_text().splitlines() if x.strip()]

def mean(xs):
    return float(np.mean(xs)) if len(xs) else float('nan')

def auc(y, score):
    y=np.asarray(y); score=np.asarray(score,float)
    pos=(y==1); n1=int(pos.sum()); n0=len(y)-n1
    if n1==0 or n0==0: return float("nan")
    try:
        from scipy.stats import rankdata
        ranks=rankdata(score, method="average")
        u=float(ranks[pos].sum() - n1*(n1+1)/2)
        return u/(n1*n0)
    except Exception:
        order=np.argsort(score)
        ranks=np.empty(len(score),float)
        ranks[order]=np.arange(1,len(score)+1)
        u=float(ranks[pos].sum() - n1*(n1+1)/2)
        return u/(n1*n0)

def exact_binom_two_sided(k,n,p=.5):
    from math import comb
    if n==0:return 1.0
    pk=comb(n,k)*(p**k)*((1-p)**(n-k))
    probs=[comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(n+1)]
    return min(1.0,sum(x for x in probs if x <= pk+1e-15))

def bootstrap_auc_diff(y,a,b,B=10000,seed=0):
    rng=np.random.default_rng(seed); y=np.asarray(y);a=np.asarray(a);b=np.asarray(b); n=len(y)
    vals=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        if len(set(y[idx]))<2: continue
        vals.append(auc(y[idx],a[idx])-auc(y[idx],b[idx]))
    return {'estimate':auc(y,a)-auc(y,b),'ci95':[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))], 'p_two_sided':float(2*min(np.mean(np.array(vals)<=0),np.mean(np.array(vals)>=0)))}

def tfidf_similarity(texts, analyzer='word'):
    # sklearn preferred; fallback simple sets
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        if analyzer=='word': vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,stop_words='english')
        else: vec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1)
        X=vec.fit_transform(texts)
        return (X@X.T).toarray()
    except Exception:
        sets=[set(t.lower().split()) for t in texts]
        M=np.zeros((len(texts),len(texts)))
        for i in range(len(texts)):
            for j in range(len(texts)):
                u=sets[i]|sets[j]; M[i,j]=len(sets[i]&sets[j])/len(u) if u else 0
        return M

def perm_similarity(rows, label_key, M, B=100000,seed=1):
    # Compare cross-carrier same-label vs different-label similarity. Permute labels within each carrier.
    carriers=sorted(set(r['carrier'] for r in rows)); inds={c:[i for i,r in enumerate(rows) if r['carrier']==c] for c in carriers}
    labels=[r[label_key] for r in rows]
    pairs=[(i,j) for i in range(len(rows)) for j in range(i+1,len(rows)) if rows[i]['carrier']!=rows[j]['carrier']]
    def stat(lab):
        same=[M[i,j] for i,j in pairs if lab[i]==lab[j]]; diff=[M[i,j] for i,j in pairs if lab[i]!=lab[j]]
        return mean(same)-mean(diff),mean(same),mean(diff),len(same),len(diff)
    obs=stat(labels)
    rng=random.Random(seed); ge=0; vals=[]
    for _ in range(B):
        lp=labels.copy()
        for c in carriers:
            ids=inds[c]; z=[lp[i] for i in ids]; rng.shuffle(z)
            for i,v in zip(ids,z): lp[i]=v
        s=stat(lp)[0]; vals.append(s); ge += (s>=obs[0]-1e-15)
    return {'delta':obs[0],'same_mean':obs[1],'different_mean':obs[2],'n_same_pairs':obs[3],'n_different_pairs':obs[4], 'perm_p_greater':(ge+1)/(B+1), 'perm_null_mean':mean(vals), 'perm_null_sd':statistics.pstdev(vals)}

out={}
# 1 free-form comparator cross-carrier text signal
comp=loadjl('comparator_tiers_v1_raw.jsonl')
texts=[r['report'] for r in comp]
free={}
for analyzer in ['word','char']:
    M=tfidf_similarity(texts,analyzer)
    free[f'target_{analyzer}']=perm_similarity(comp,'concept',M,B=1500,seed=10 if analyzer=='word' else 11)
    free[f'model_pick_{analyzer}']=perm_similarity(comp,'model_forced_pick',M,B=1500,seed=12 if analyzer=='word' else 13)
# Reader pick relationship to true target and model pick
free['t1_matches_target']=mean([r['t1_reader_pick']==r['concept'] for r in comp])
free['t1_matches_model_forced_pick']=mean([r['t1_reader_pick']==r['model_forced_pick'] for r in comp])
free['model_forced_accuracy']=mean([r['model_forced_correct'] for r in comp])
free['t1_accuracy']=mean([r['t1_reader_correct'] for r in comp])
free['report_mentions_target']=mean([r['report_mentions_target'] for r in comp])
free['model_pick_counts']=Counter(r['model_forced_pick'] for r in comp)
free['t1_pick_counts']=Counter(r['t1_reader_pick'] for r in comp)
out['free_form_report_signal']=free

# 2 Join zero-shot/comparator/lens depth
zero=loadjl('zero_shot_identify_v1_raw.jsonl')
model2={(r['carrier'],r['concept']):r for r in zero if r['arm']=='model_injected' and float(r['strength'])==2.0}
lens=loadjl('lens_depth_v1_raw.jsonl')
by=defaultdict(dict)
for r in lens: by[(r['carrier'],r['concept'])][(r['site'],int(r['depth']))]=bool(r['correct'])
join=[]
for key,m in model2.items():
    d=by[key]
    final_correct=[dep for dep in range(9,36) if d.get(('final',dep),False)]
    marker_correct=[dep for dep in range(9,36) if d.get(('marker',dep),False)]
    first_final=min(final_correct) if final_correct else 99
    # earliest depth from which all remaining final lenses are correct
    sustained=99
    for dep in range(9,36):
        if all(d.get(('final',x),False) for x in range(dep,36)):
            sustained=dep; break
    join.append({'carrier':key[0],'concept':key[1],'model_correct':bool(m['correct']),'model_predicted':m['predicted'], 'first_final_lens_correct':first_final,'sustained_final_lens_correct':sustained,'n_final_layers_correct':len(final_correct),'n_marker_layers_correct':len(marker_correct),'final_lens_at_35':d.get(('final',35),False),'marker_lens_at_35':d.get(('marker',35),False)})
y=[int(r['model_correct']) for r in join]
lensjoin={'n':len(join),'model_accuracy':mean(y),'rows':join}
for metric,reverse in [('first_final_lens_correct',True),('sustained_final_lens_correct',True),('n_final_layers_correct',False),('n_marker_layers_correct',False)]:
    s=np.array([r[metric] for r in join],float); sc=-s if reverse else s
    lensjoin[metric+'_auc_for_model_correct']=auc(np.array(y),sc)
    lensjoin[metric+'_mean_model_right']=mean([r[metric] for r in join if r['model_correct']])
    lensjoin[metric+'_mean_model_wrong']=mean([r[metric] for r in join if not r['model_correct']])
lensjoin['model_vs_final_lens35_cross_tab']={str(k):v for k,v in Counter((r['model_correct'],r['final_lens_at_35']) for r in join).items()}
# exact mismatches model vs lens35
out['zero_shot_lens_depth_join']=lensjoin

# 3 Reader depth cell-level relation
rd=loadjl('reader_depth_v1_raw.jsonl')
rdout={'n_rows':len(rd)}
# use layers 9..35; accuracy vector index assumed layer index
surv=[]
for r in rd:
    arr=r['reader_correct_by_layer']
    post=arr[9:]
    last=max([i+9 for i,v in enumerate(post) if v], default=8)
    n=sum(post)
    # first failure after layer9, and sustained correctness end
    first_fail=next((i+9 for i,v in enumerate(post) if not v),36)
    surv.append((int(r['model_correct']),last,n,first_fail,int(arr[35])))
y=np.array([x[0] for x in surv])
for name,idx in [('last_correct_layer',1),('n_post_layers_correct',2),('first_failure_layer',3),('reader_final_correct',4)]:
    s=np.array([x[idx] for x in surv],float)
    rdout[name+'_auc_for_model_correct']=auc(y,s)
    rdout[name+'_mean_model_right']=float(np.mean(s[y==1])); rdout[name+'_mean_model_wrong']=float(np.mean(s[y==0]))
# pair-level model twin vs reader twin by layer
groups=defaultdict(list)
for r in rd:
    # same prompt sha + carrier + concept should twin signs, but mapping cells differ prompts
    groups[(r['carrier'],r['concept'],r['prompt_sha256'])].append(r)
pairs=[g for g in groups.values() if len(g)==2]
rdout['n_twin_pairs']=len(pairs)
rdout['model_twin_accuracy']=mean([all(x['model_correct'] for x in g) for g in pairs])
rdout['reader_twin_by_layer']={str(L):mean([all(x['reader_correct_by_layer'][L] for x in g) for g in pairs]) for L in range(9,36)}
rdout['model_right_reader_wrong_by_layer']={str(L):sum(all(x['model_correct'] for x in g) and not all(x['reader_correct_by_layer'][L] for x in g) for g in pairs) for L in range(9,36)}
out['reader_depth_behavior_link']=rdout

# 4 Zero-shot per-concept difficulty and lens correction
zrows=[r for r in zero if r['arm']=='model_injected']
lrows=[r for r in zero if r['arm']=='lens_injected']
zk={(r['carrier'],r['concept'],float(r['strength'])):r for r in zrows}; lk={(r['carrier'],r['concept'],float(r['strength'])):r for r in lrows}
zout={}
zout['model_per_concept']={c:mean([r['correct'] for r in zrows if r['concept']==c]) for c in sorted(set(r['concept'] for r in zrows))}
zout['model_per_carrier']={c:mean([r['correct'] for r in zrows if r['carrier']==c]) for c in sorted(set(r['carrier'] for r in zrows))}
zout['model_per_strength']={str(s):mean([r['correct'] for r in zrows if float(r['strength'])==s]) for s in sorted(set(float(r['strength']) for r in zrows))}
zout['model_lens_agreement']=mean([zk[k]['predicted']==lk[k]['predicted'] for k in zk])
zout['lens_correct_when_model_wrong']=mean([lk[k]['correct'] for k in zk if not zk[k]['correct']])
zout['n_model_wrong']=sum(not r['correct'] for r in zrows)
zout['wrong_prediction_counts']=Counter(r['predicted'] for r in zrows if not r['correct'])
zout['correct_prediction_counts']=Counter(r['predicted'] for r in zrows if r['correct'])
# consistency across carrier: for each concept,strength, fraction pairs same model predicted
cons=[]
for c in sorted(set(r['concept'] for r in zrows)):
 for s in sorted(set(float(r['strength']) for r in zrows)):
  ps=[r['predicted'] for r in zrows if r['concept']==c and float(r['strength'])==s]
  cons.append(max(Counter(ps).values())/len(ps))
zout['mean_cross_carrier_modal_prediction_fraction']=mean(cons)
out['zero_shot_error_structure']=zout

# 5 self knowledge confirm incremental value
sk=loadjl('self_knowledge_confirm_v1_raw.jsonl')
test=[r for r in sk if 'probe_score' in r]
y=np.array([int(r['correct']) for r in test])
margin=np.array([float(r['margin']) for r in test])
probe=np.array([float(r['probe_score']) for r in test])
verbal=np.array([float(r['verbal_yes_minus_no']) for r in test])
size=np.array([float(r['a']*r['b']) for r in test])
skout={'n':len(test),'accuracy':mean(y)}
for nm,s in [('margin',margin),('probe',probe),('verbal_logit',verbal),('product_size',size)]: skout['auc_'+nm]=auc(y,s)
skout['auc_diff_margin_minus_probe']=bootstrap_auc_diff(y,margin,probe,B=1500,seed=21)
skout['auc_diff_margin_minus_verbal']=bootstrap_auc_diff(y,margin,verbal,B=1500,seed=22)
skout['correlations']={}
for a_nm,a in [('margin',margin),('probe',probe),('verbal',verbal),('size',size)]:
 for b_nm,b in [('margin',margin),('probe',probe),('verbal',verbal),('size',size)]:
  if a_nm<b_nm: skout['correlations'][a_nm+'__'+b_nm]=float(np.corrcoef(a,b)[0,1])
# repeated 5-fold CV logistic incremental metrics
try:
 from sklearn.model_selection import RepeatedStratifiedKFold
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 from sklearn.pipeline import make_pipeline
 specs={'margin':[margin],'probe':[probe],'margin_probe':[margin,probe],'margin_probe_verbal':[margin,probe,verbal],'margin_size':[margin,size],'margin_probe_size':[margin,probe,size]}
 Xs={k:np.column_stack(v) for k,v in specs.items()}
 cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=5,random_state=23)
 scores={k:[] for k in specs}
 for tr,te in cv.split(np.zeros(len(y)),y):
  for k,X in Xs.items():
   m=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,max_iter=1000))
   m.fit(X[tr],y[tr]); pr=m.predict_proba(X[te])[:,1]; scores[k].append(auc(y[te],pr))
 skout['repeated_cv_auc']={k:{'mean':mean(v),'sd':statistics.pstdev(v)} for k,v in scores.items()}
 diffs=np.array(scores['margin_probe'])-np.array(scores['margin'])
 skout['cv_margin_probe_minus_margin']={'mean':float(diffs.mean()),'ci95':[float(np.quantile(diffs,.025)),float(np.quantile(diffs,.975))]}
except Exception as e:
 skout['cv_error']=repr(e)
out['self_knowledge_incremental']=skout

# convert counters recursively
class Enc(json.JSONEncoder):
 def default(self,o):
  if isinstance(o,Counter):return dict(o)
  if isinstance(o,np.integer):return int(o)
  if isinstance(o,np.floating):return float(o)
  if isinstance(o,np.bool_):return bool(o)
  return super().default(o)
OUT.write_text(json.dumps(out,indent=2,cls=Enc))
print(OUT)
# concise
for k,v in out.items():
 print('\n##',k)
 if k=='free_form_report_signal':
  print(json.dumps(v,indent=2,cls=Enc))
 elif k=='zero_shot_lens_depth_join':
  print({x:y for x,y in v.items() if x!='rows'})
 else: print(json.dumps(v,indent=2,cls=Enc)[:7000])
