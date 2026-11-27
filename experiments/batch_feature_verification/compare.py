import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import sys, gzip, json, time, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'src'
OUT=Path(__file__).resolve().parent/'rerun_results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(SRC))
import model, interest_profile
original=model.run_cv
def compressed(path,mode='r',**kwargs):
    if str(path).endswith('.gz'):
        return gzip.open(path,'rt' if mode=='r' else mode,**kwargs)
    return open(path,mode,**kwargs)
interest_profile.open=compressed
report={'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'threads':1},'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.glob('*.py')},'results':{}}
def run(candidates,features,folds,k,params,kparams,content,locale_on=False,calendar_on=False,relative_on=False):
    report['model_parameters']=params;report['kmeans_parameters']=kparams
    candidates[['user','event']].assign(fold=folds).to_csv(OUT/'folds.csv',index=False)
    for name,relative in [('without_batch',False),('with_batch',True)]:
        print('START',name,report['environment'],flush=True)
        start=time.time()
        scores,oof=original(candidates,features,folds,k,params,kparams,content,locale_on,calendar_on,relative)
        scores.to_csv(OUT/(name+'_fold_scores.csv'),index=False)
        oof.to_csv(OUT/(name+'_oof.csv'),index=False)
        report['results'][name]={'mean':float(scores.map200.mean()),'folds':scores.to_dict('records'),'seconds':time.time()-start}
        (OUT/'results.json').write_text(json.dumps(report,indent=2))
        print('RESULT',name,report['results'][name]['mean'],flush=True)
    return scores,oof
model.run_cv=run
source=(SRC/'data.py').read_text()
for name in ['events','event_attendees','user_friends']:
    if not (ROOT/'data/raw'/(name+'.csv')).exists():
        source=source.replace('"'+name+'.csv"','"'+name+'.csv.gz"')
with threadpool_limits(limits=1):
    exec(compile(source,str(SRC/'data.py'),'exec'),{'__name__':'__main__','__file__':str(SRC/'data.py')})
assert all(hashlib.sha256((SRC/n).read_bytes()).hexdigest()==v for n,v in report['source_hashes'].items())
print('COMPLETE; source unchanged',flush=True)
