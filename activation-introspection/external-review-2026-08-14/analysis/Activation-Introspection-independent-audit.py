from __future__ import annotations
import json, re, hashlib, statistics, math
from pathlib import Path
from collections import defaultdict, Counter
ROOT=Path('/mnt/data/activation_audit/a3')

def loadj(path): return json.loads((ROOT/path).read_text())
def loadjl(path):
    with (ROOT/path).open() as f: return [json.loads(x) for x in f if x.strip()]
def mean(xs): return sum(xs)/len(xs) if xs else float('nan')
def rate(rows,key='correct'): return mean([1.0 if r[key] else 0.0 for r in rows])
def sha(path): return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def cellparts(cell):
    m=re.fullmatch(r'o(\d+)m(\d+)q([+-]?\d+)',cell)
    if not m: raise ValueError(cell)
    return tuple(map(int,m.groups()))

def both_correct(groups, field='correct'):
    ok=[]; badsizes=[]
    for k,rs in groups.items():
        if len(rs)!=2: badsizes.append((k,len(rs)))
        ok.append(len(rs)==2 and all(r[field] is True for r in rs))
    return mean(ok), badsizes

def codebook():
    rows=loadjl(Path('results/codebook_icl_confirm_v2_raw.jsonl'))
    out={}
    conds=rows[0]['condition_scores'].keys()
    for c in conds:
        out[c]={'row_acc':mean([r['condition_scores'][c]['correct'] for r in rows]),
                'format':mean([r['condition_scores'][c]['format_ok'] for r in rows]),
                'label_mass':mean([r['condition_scores'][c]['label_mass'] for r in rows])}
        g=defaultdict(list); gm=defaultdict(list)
        for r in rows:
            o,m,q=cellparts(r['cell_id'])
            sc=r['condition_scores'][c]
            rr={'correct':sc['correct']}
            g[(r['concept'],r['carrier_id'],o,m)].append(rr)
            gm[(r['concept'],r['carrier_id'],o,q)].append(rr)
        out[c]['twin_pair'],bad=both_correct(g)
        out[c]['mapping_flip'],bad2=both_correct(gm)
        assert not bad and not bad2
    # visible prompt identity across q twins
    pg=defaultdict(set)
    for r in rows:
        o,m,q=cellparts(r['cell_id']); pg[(r['concept'],r['carrier_id'],o,m)].add(r['prompt_sha256'])
    prompt_twin_identical=all(len(v)==1 for v in pg.values())
    # correct labels must be opposite within q twins
    lg=defaultdict(set)
    for r in rows:
        o,m,q=cellparts(r['cell_id']); lg[(r['concept'],r['carrier_id'],o,m)].add(r['correct_label'])
    labels_opposite=all(v=={'Q','K'} for v in lg.values())
    # condition direction hashes unique as expected
    byconcept={}
    for concept in sorted(set(r['concept'] for r in rows)):
        rs=[r for r in rows if r['concept']==concept]
        byconcept[concept]={c:set(r['condition_scores'][c]['direction_sha256'] for r in rs) for c in conds}
    random_hashes={next(iter(v['random'])) for v in byconcept.values()}
    target_hashes={next(iter(v['target'])) for v in byconcept.values()}
    out['_validation']={'rows':len(rows),'prompt_twin_identical':prompt_twin_identical,
                        'labels_opposite':labels_opposite,'random_unique_concepts':len(random_hashes),
                        'target_unique_concepts':len(target_hashes),
                        'target_testonly_same_hash':all(v['target']==v['test_only'] for v in byconcept.values())}
    return out

def report_training():
    seeds=[]
    for s in range(4):
        rows=loadjl(Path(f'results/report_training_v3_seed{s}_raw.jsonl'))
        seedout={}
        for arm in sorted(set(r['arm'] for r in rows)):
            for cond in ['target','random','shuffled']:
                rs=[r for r in rows if r['arm']==arm and r['condition']==cond]
                g=defaultdict(list)
                for r in rs: g[(r['concept'],r['carrier_sha256'])].append(r)
                twin,bad=both_correct(g); assert not bad
                seedout[f'{arm}/{cond}']={'row_acc':rate(rs),'twin_pair':twin,'format':mean([r['format_ok'] for r in rs]),'label_mass':mean([r['label_mass'] for r in rs])}
        # raw semantic validation
        random_rs=[r for r in rows if r['condition']=='random']
        seedout['_validation']={
            'rows':len(rows),
            'random_rows_have_expected_label':all(r['expected_label'] in {'Q','K'} for r in random_rs),
            'random_rows_have_binary_correct':all(isinstance(r['correct'],bool) for r in random_rs),
            'clean_rows_unscored':all(r['expected_label'] is None and r['correct'] is None for r in rows if r['condition']=='clean'),
        }
        seeds.append(seedout)
    return seeds

def remap():
    seeds=[]
    for s in range(3):
        rows=loadjl(Path(f'results/remap_training_v2_seed{s}_raw.jsonl'))
        seedout={}
        for arm in ['base','fixed','remap']:
          for cond in ['target','random']:
            strengths=sorted(set(r['strength'] for r in rows if r['arm']==arm and r['condition']==cond))
            for st in strengths:
                rs=[r for r in rows if r['arm']==arm and r['condition']==cond and r['strength']==st]
                gt=defaultdict(list); gm=defaultdict(list)
                for r in rs:
                    o,m,q=cellparts(r['cell_id'])
                    gt[(r['concept'],r['carrier_sha256'],o,m,st)].append(r)
                    gm[(r['concept'],r['carrier_sha256'],o,q,st)].append(r)
                twin,bad=both_correct(gt); mf,bad2=both_correct(gm); assert not bad and not bad2
                seedout[f'{arm}/{cond}@{st}']={'row_acc':rate(rs),'twin_pair':twin,'mapping_flip':mf,'format':mean([r['format_ok'] for r in rs]),'label_mass':mean([r['label_mass'] for r in rs])}
        random_rs=[r for r in rows if r['condition']=='random']
        # q twins: identical prompt sha, opposite correct labels
        p=defaultdict(set); lab=defaultdict(set)
        for r in rows:
            o,m,q=cellparts(r['cell_id'])
            key=(r['arm'],r['condition'],r['strength'],r['concept'],r['carrier_sha256'],o,m)
            p[key].add(r['prompt_sha256']); lab[key].add(r['correct_label'])
        seedout['_validation']={
          'rows':len(rows),
          'random_rows_have_correct_label':all(r['correct_label'] in {'Q','K'} for r in random_rs),
          'random_rows_scored':all(isinstance(r['correct'],bool) for r in random_rs),
          'q_twin_visible_prompt_identical':all(len(v)==1 for v in p.values()),
          'q_twin_labels_opposite':all(v=={'Q','K'} for v in lab.values()),
          'base_identical_across_seeds_signature': hashlib.sha256('\n'.join(json.dumps({k:v for k,v in r.items() if k!='seed'},sort_keys=True) for r in rows if r['arm']=='base').encode()).hexdigest()
        }
        seeds.append(seedout)
    return seeds

def heldout():
    rows=loadjl(Path('results/heldout_semantic_v1_raw.jsonl'))
    out={}
    for arm in sorted(set(r['arm'] for r in rows)):
        rs=[r for r in rows if r['arm']==arm]
        g=defaultdict(list)
        for r in rs: g[(r['pair'],r['carrier_sha'],r['cell_base'])].append(r)
        mt,bad=both_correct(g,'model_correct'); assert not bad
        rc,bad=both_correct(g,'reader_centroid_cosine_correct'); assert not bad
        reu,bad=both_correct(g,'reader_centroid_euclidean_correct'); assert not bad
        rsh,bad=both_correct(g,'reader_shuffled_labels_correct'); assert not bad
        out[arm]={
          'row_model':mean([r['model_correct'] for r in rs]),'twin_model':mt,
          'row_reader_cos':mean([r['reader_centroid_cosine_correct'] for r in rs]),'twin_reader_cos':rc,
          'row_reader_euc':mean([r['reader_centroid_euclidean_correct'] for r in rs]),'twin_reader_euc':reu,
          'twin_reader_shuffled':rsh,
          'format':mean([r['model_format_ok'] for r in rs])
        }
    # sanity: all twins two rows opposite q signs
    v=defaultdict(set)
    for r in rows: v[(r['arm'],r['pair'],r['carrier_sha'],r['cell_base'])].add(r['query_sign'])
    out['_validation']={'rows':len(rows),'all_twins_opposite_signs':all(x=={-1,1} for x in v.values())}
    return out

def prompt_clash(path, stance=False):
    rows=loadjl(Path(path)); out={}
    if not stance:
      for inst in sorted(set(r['instruction_stance'] for r in rows)):
       for car in sorted(set(r['carrier_stance'] for r in rows)):
        rs=[r for r in rows if r['instruction_stance']==inst and r['carrier_stance']==car]
        g=defaultdict(list)
        for r in rs: g[(r['pair'],r['cell_base'])].append(r)
        twin,bad=both_correct(g); assert not bad
        out[f'{inst}/{car}']={'row':rate(rs),'twin':twin,'mean_margin':mean([r['margin'] for r in rs])}
    else:
      for inst in sorted(set(r['instruction'] for r in rows)):
       for car in sorted(set(r['carrier_stance'] for r in rows)):
        rs=[r for r in rows if r['instruction']==inst and r['carrier_stance']==car]
        g=defaultdict(list)
        for r in rs: g[(r['pair'],r['cell_base'])].append(r)
        twin,bad=both_correct(g); assert not bad
        out[f'{inst}/{car}']={'row':rate(rs),'twin':twin,'mean_margin':mean([r['margin'] for r in rs])}
    return out

def integrity():
    out={'manifest_count':0,'raw_sha_mismatches':[],'protocol_links':[]}
    for mp in sorted(ROOT.glob('results/*manifest.json')):
        d=json.loads(mp.read_text()); out['manifest_count']+=1
        if d.get('raw') and d.get('raw_sha256'):
            rp=ROOT/'results'/d['raw']
            got=hashlib.sha256(rp.read_bytes()).hexdigest() if rp.exists() else None
            if got!=d['raw_sha256']: out['raw_sha_mismatches'].append((mp.name,d['raw'],d['raw_sha256'],got))
        psha=d.get('config',{}).get('protocol_sha256')
        if psha:
            matches=[]
            for pp in ROOT.glob('results/*protocol*.json'):
                if hashlib.sha256(pp.read_bytes()).hexdigest()==psha: matches.append(pp.name)
            out['protocol_links'].append((mp.name,psha,matches))
    return out

res={'integrity':integrity(),'codebook':codebook(),'report_training':report_training(),'remap':remap(),'heldout':heldout(),'prompt_clash_v2':prompt_clash('results/prompt_clash_v2_raw.jsonl'),'instruction_stance':prompt_clash('results/instruction_stance_v1_raw.jsonl',True)}
Path('/mnt/data/activation_audit/independent_metrics.json').write_text(json.dumps(res,indent=2,sort_keys=True))
print(json.dumps(res['codebook'],indent=2))
print('\nREPORT trained targets by seed')
for i,s in enumerate(res['report_training']): print(i,s['trained/target'], 'random',s['trained/random'])
print('\nREMAP headline')
for i,s in enumerate(res['remap']):
 print('seed',i)
 for k in ['base/target@0.5','fixed/target@0.5','remap/target@0.5','base/target@0.25','fixed/target@0.25','remap/target@0.25','base/target@0.15','fixed/target@0.15','remap/target@0.15','base/random@0.5','fixed/random@0.5','remap/random@0.5']:
  print(k,s[k])
print('\nHELDOUT')
print(json.dumps(res['heldout'],indent=2))
print('\nINTEGRITY mismatches',res['integrity']['raw_sha_mismatches'])
print('protocol links missing',[x for x in res['integrity']['protocol_links'] if not x[2]])
