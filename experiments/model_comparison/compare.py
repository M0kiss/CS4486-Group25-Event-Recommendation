"""Four-model comparison using the project feature pipeline.
AI-assisted comparison harness; original settings and saved results are documented in README.md.
"""
from pathlib import Path
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='2'
import sys, runpy, json, time, warnings, hashlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'src'
OUT=ROOT/'rerun_results'
OUT.mkdir(exist_ok=True)
BASELINE='RF+LR reference adaptation'
BLEND_WEIGHTS=(0.69, 0.57)
sys.path.insert(0,str(SOURCE))
import model as original_model
from feature import cluster_feature,locale_feature,calendar_feature,query_relative_feature
from validation import ap

def compare(cddts,features,folds,k,model_parameters,kmeans_parameters,content,locale_on=False,calendar_on=False,relative_on=False):
    assert not cddts.duplicated(['user','event']).any()
    assert cddts.assign(fold=folds).groupby('user')['fold'].nunique().max()==1
    assert ap([1,2],[1,3,2],200)==(1+2/3)/2
    assert ap([], [1],200)==0
    labels=cddts.interested
    shared=features.copy()
    if calendar_on: shared=pd.concat([shared,calendar_feature(cddts)],axis=1)
    if relative_on: shared=pd.concat([shared,query_relative_feature(cddts,features)],axis=1)
    rows=[]; user_rows=[]
    oof=cddts[['user','event','interested']].copy();oof['fold']=folds
    params={'HistGradientBoosting':model_parameters,'RandomForest':dict(n_estimators=300,min_samples_leaf=5,max_features='sqrt',random_state=25,n_jobs=2),'MLP':dict(hidden_layer_sizes=(64,32),alpha=0.01,max_iter=150,early_stopping=False,random_state=25)}
    manifest={'seed':25,'folds':5,'metric':'MAP@200; users with no positive labels score zero; mean of fold means','models':params,'versions':{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.py')},'kmeans':kmeans_parameters,'feature_switches':{'locale':locale_on,'calendar':calendar_on,'relative':relative_on},'notes':['No tuning search; fixed candidate configurations. Same CV used for exploratory comparison, not an independent final test.','RF uses training-fold median imputation; MLP uses training-fold imputation and scaling.','Auxiliary attendance snapshots have unverified temporal availability.']}
    logistic_parameters = dict(C=0.03, l1_ratio=1.0, solver='liblinear',
                               max_iter=1000, random_state=25)
    manifest['reference_adaptation'] = {
        'source': 'https://github.com/andreiolariu/kaggle-event-recommendation/blob/master/model.py',
        'random_forest': 'Reuses the standalone RandomForest in this comparison',
        'logistic_regression': logistic_parameters,
        'preprocessing': 'Training-fold median imputation and scaling for logistic regression',
        'weights': dict(zip(('RandomForest', 'LogisticRegression'), BLEND_WEIGHTS)),
        'features': 'Group feature set for both branches; source used separate feature subsets',
        'timing': 'Blend timing includes the reused RF fit/predict time',
    }
    (OUT/'config.json').write_text(json.dumps(manifest,indent=2))
    cddts[['user','event']].assign(fold=folds).to_csv(OUT/'folds.csv',index=False)
    for fid in sorted(folds.unique()):
        tr=folds!=fid;va=~tr
        assert set(cddts.loc[tr,'user']).isdisjoint(cddts.loc[va,'user'])
        X=shared.copy()
        if content is not None:X=pd.concat([X,cluster_feature(cddts,content,tr,kmeans_parameters)],axis=1)
        if locale_on:X=pd.concat([X,locale_feature(cddts,tr)],axis=1)
        assert not {'interested','not_interested','user','event'} & set(X.columns)
        assert np.isfinite(X.fillna(0).to_numpy(dtype=float)).all()
        (OUT/f'feature_columns_fold_{fid}.json').write_text(json.dumps(list(X.columns)))
        models={'HistGradientBoosting':HistGradientBoostingClassifier(**params['HistGradientBoosting']), 'RandomForest':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestClassifier(**params['RandomForest'])), 'MLP':make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),MLPClassifier(**params['MLP']))}
        models[BASELINE] = make_pipeline(
            SimpleImputer(strategy='median', keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(**logistic_parameters),
        )
        fold_scores = {}
        rf_seconds = 0.0
        for name,est in models.items():
            print(f'START fold={fid} model={name}',flush=True)
            start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                with threadpool_limits(limits=2):
                    est.fit(X.loc[tr],labels.loc[tr]);scores=est.predict_proba(X.loc[va])[:,1]
            elapsed=time.perf_counter()-start
            if name == 'RandomForest':
                fold_scores['RandomForest'] = scores
                rf_seconds = elapsed
            if name == BASELINE:
                oof.loc[va, 'LogisticRegression_component'] = scores
                scores = (BLEND_WEIGHTS[0] * fold_scores['RandomForest']
                          + BLEND_WEIGHTS[1] * scores)
                elapsed += rf_seconds
            oof.loc[va,name]=scores
            v=oof.loc[va];aps=[]
            for user,g in v.groupby('user'):
                score=ap(g.loc[g.interested==1,'event'].tolist(),g.sort_values([name,'event'],ascending=[False,True]).event.tolist(),k)
                aps.append(score);user_rows.append(dict(model=name,fold=int(fid),user=int(user),ap=score))
            row=dict(model=name,fold=int(fid),users=len(aps),map200=float(np.mean(aps)),seconds=elapsed,warnings=' | '.join(str(w.message) for w in caught))
            rows.append(row);print(row,flush=True)
            pd.DataFrame(rows).to_csv(OUT/'fold_scores.csv',index=False)
            pd.DataFrame(user_rows).to_csv(OUT/'user_scores.csv',index=False)
            oof.to_csv(OUT/'oof.csv',index=False)
    results=pd.DataFrame(rows)
    summary=results.groupby('model').agg(mean_map200=('map200','mean'),fold_sd=('map200','std'),total_fit_predict_seconds=('seconds','sum'))
    summary['user_weighted_map200']=pd.DataFrame(user_rows).groupby('model')['ap'].mean()
    summary.sort_values('mean_map200',ascending=False).to_csv(OUT/'summary.csv')
    print(summary.to_string(),flush=True)
    return results.loc[results.model=='HistGradientBoosting'],oof

if __name__=='__main__':
    original_model.run_cv=compare
    runpy.run_path(str(SOURCE/'data.py'),run_name='__main__')
