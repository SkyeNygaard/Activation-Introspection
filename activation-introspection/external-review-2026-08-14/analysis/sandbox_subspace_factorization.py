import json, torch, numpy as np
from pathlib import Path
from itertools import product
BASE=Path('/mnt/data/activation_introspection/extract3/results')
FILES={
 'qwen05b':(BASE/'retained_test_qwen05b_v2_raw.acts.pt',None),
 'qwen15b':(BASE/'retained_dev_qwen15b_raw.acts.pt',1.0),
 'qwen3b':(BASE/'retained_dev_qwen3b_raw.acts.pt',1.0),
}
def basis(M,tol=1e-7):
    # rows are effect vectors; return d x rank orthonormal
    if M.numel()==0:return torch.zeros((M.shape[1],0))
    U,S,Vh=torch.linalg.svd(M.float(),full_matrices=False)
    r=int((S>S.max()*tol).sum()) if S.numel() else 0
    return Vh[:r].T.contiguous()
def proj_off(X,Q):
    return X-X@Q@Q.T if Q.shape[1] else X

def cosine_classify(X,labels,trainmask,testmask):
    labs=sorted(set(labels)); cents={l:X[[i for i,m in enumerate(trainmask) if m and labels[i]==l]].mean(0) for l in labs}
    C=torch.stack([cents[l] for l in labs]); C=C/(C.norm(dim=1,keepdim=True)+1e-9)
    ids=[i for i,m in enumerate(testmask) if m]; Z=X[ids];Z=Z/(Z.norm(dim=1,keepdim=True)+1e-9)
    pred=(Z@C.T).argmax(1)
    return sum(labs[int(j)]==labels[i] for i,j in zip(ids,pred))/len(ids)

def analyze(path,strength):
    b=torch.load(path,map_location='cpu',weights_only=False); A=b['acts'].float(); idx=b['index']
    if strength is not None:
      keep=[i for i,m in enumerate(idx) if float(m.get('strength',1))==strength];A=A[keep];idx=[idx[i] for i in keep]
    Xall=A[:,-1]
    out={}
    for arm in ['target','random','shuffled']:
      ids=[i for i,m in enumerate(idx) if m['arm']==arm]; X=Xall[ids]; meta=[idx[i] for i in ids]
      mu=X.mean(0); Xc=X-mu
      concepts=sorted({m['concept'] for m in meta});depths=sorted({m['inject_layer'] for m in meta});cars=sorted({m['carrier_id'] for m in meta})
      CE=torch.stack([X[[j for j,m in enumerate(meta) if m['concept']==c]].mean(0)-mu for c in concepts])
      DE=torch.stack([X[[j for j,m in enumerate(meta) if m['inject_layer']==d]].mean(0)-mu for d in depths])
      AE=torch.stack([X[[j for j,m in enumerate(meta) if m['carrier_id']==c]].mean(0)-mu for c in cars])
      Qc,Qd,Qa=map(basis,[CE,DE,AE])
      sv=torch.linalg.svdvals(Qc.T@Qd) if Qc.shape[1] and Qd.shape[1] else torch.tensor([])
      # fraction of concept effect energy explained by depth / vice versa
      ce_depth=float((CE@Qd).pow(2).sum()/CE.pow(2).sum()) if Qd.shape[1] else 0
      de_concept=float((DE@Qc).pow(2).sum()/DE.pow(2).sum()) if Qc.shape[1] else 0
      # leave one depth+carrier jointly for concept classification before/after residualizing global depth subspace
      def concept_cv(Z):
        correct=total=0
        for hd in depths:
          for hc in cars:
            train=[m['inject_layer']!=hd and m['carrier_id']!=hc for m in meta]
            test=[m['inject_layer']==hd and m['carrier_id']==hc for m in meta]
            if not any(train) or not any(test):continue
            a=cosine_classify(Z,[m['concept'] for m in meta],train,test); n=sum(test);correct+=a*n;total+=n
        return correct/total
      # leave concept + carrier for depth classification
      def depth_cv(Z):
        correct=total=0
        for hcpt in concepts:
          for hcar in cars:
            train=[m['concept']!=hcpt and m['carrier_id']!=hcar for m in meta]
            test=[m['concept']==hcpt and m['carrier_id']==hcar for m in meta]
            a=cosine_classify(Z,[m['inject_layer'] for m in meta],train,test);n=sum(test);correct+=a*n;total+=n
        return correct/total
      out[arm]={
        'ranks':{'concept':Qc.shape[1],'depth':Qd.shape[1],'carrier':Qa.shape[1]},
        'concept_depth_principal_cosines':[float(x) for x in sv],
        'concept_depth_max_cosine':float(sv.max()) if len(sv) else None,
        'concept_depth_mean_sq_cosine':float((sv**2).mean()) if len(sv) else None,
        'concept_effect_energy_in_depth_subspace':ce_depth,
        'depth_effect_energy_in_concept_subspace':de_concept,
        'concept_cv_original':concept_cv(Xc),
        'concept_cv_depth_residualized':concept_cv(proj_off(Xc,Qd)),
        'depth_cv_original':depth_cv(Xc),
        'depth_cv_concept_residualized':depth_cv(proj_off(Xc,Qc)),
        'chance_concept':1/len(concepts),'chance_depth':1/len(depths),
      }
    return out
res={k:analyze(*v) for k,v in FILES.items()}
print(json.dumps(res,indent=2))
open('/mnt/data/activation_introspection/sandbox_subspace_factorization_results.json','w').write(json.dumps(res,indent=2)+'\n')
