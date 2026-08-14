import json,os,statistics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid
R='/mnt/data/activation_introspection/extract3/results'
rows=[json.loads(x) for x in open(f'{R}/comparator_tiers_v1_raw.jsonl')]
carriers=sorted(set(r['carrier'] for r in rows)); labels=sorted(set(r['concept'] for r in rows))
print('carriers',carriers,'labels',labels)
configs=[('word',(1,2),'word'),('char',(3,5),'char')]
out={}
for name,ng,an in configs:
 for modelname in ['logreg','svm','centroid']:
  fold=[]; preds=[]
  for testc in carriers:
   tr=[r for r in rows if r['carrier']!=testc]; te=[r for r in rows if r['carrier']==testc]
   vec=TfidfVectorizer(analyzer=an,ngram_range=ng,min_df=1,sublinear_tf=True)
   X=vec.fit_transform([r['report'] for r in tr]); Xt=vec.transform([r['report'] for r in te])
   y=[r['concept'] for r in tr]; yt=[r['concept'] for r in te]
   if modelname=='logreg': clf=LogisticRegression(C=1.0,max_iter=5000)
   elif modelname=='svm': clf=LinearSVC(C=0.5)
   else: clf=NearestCentroid(metric='euclidean')
   clf.fit(X.toarray() if modelname=='centroid' else X,y)
   yp=clf.predict(Xt.toarray() if modelname=='centroid' else Xt)
   fold.append(accuracy_score(yt,yp)); preds.extend(zip([testc]*len(te),yt,yp))
  out[f'{name}_{modelname}']={'fold_acc':fold,'mean':statistics.mean(fold),'preds':preds}
# exact nearest report cosine across other carriers
from sklearn.metrics.pairwise import cosine_similarity
for name,ng,an in configs:
 fold=[]
 for testc in carriers:
  tr=[r for r in rows if r['carrier']!=testc]; te=[r for r in rows if r['carrier']==testc]
  vec=TfidfVectorizer(analyzer=an,ngram_range=ng,min_df=1,sublinear_tf=True)
  X=vec.fit_transform([r['report'] for r in tr]); Xt=vec.transform([r['report'] for r in te])
  S=cosine_similarity(Xt,X); yp=[tr[i]['concept'] for i in S.argmax(axis=1)]
  fold.append(accuracy_score([r['concept'] for r in te],yp))
 out[f'{name}_1nn']={'fold_acc':fold,'mean':statistics.mean(fold)}
print(json.dumps(out,indent=2))
open('/mnt/data/activation_introspection/sandbox_freeform_cv_results.json','w').write(json.dumps(out,indent=2)+'\n')
