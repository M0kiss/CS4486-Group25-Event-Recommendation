
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from validation import ap
from feature import cluster_feature,locale_feature,calendar_feature,query_relative_feature

def run_cv(cddts,features,folds,k,model_parameters,kmeans_parameters,content,locale_on=False,calendar_on=False,relative_on=False):
    '''模型训练&评分'''
    labels=cddts['interested']

    #outoffold用于验证以某fold为测试集下的user得分
    oof= cddts[["user", "event", "interested"]].copy()
    oof["fold"]= folds
    oof["score"]= np.nan

    fold_results=[]

    calendar_features = None
    if calendar_on:
        calendar_features = calendar_feature(cddts)

    relative_features= None
    if relative_on: 
        relative_features=query_relative_feature(cddts,features)  

    for fold_id in sorted(folds.unique()): #各fold轮流作验证集
        trainmask= (folds != fold_id)
        validmask= (folds == fold_id)

        #事件分类
        fold_features = features
        if content is not None:
            clusters= cluster_feature(cddts,content,trainmask,kmeans_parameters)
            fold_features= pd.concat([features,clusters],axis=1) #拼接所属类别
        if locale_on:
            locales= locale_feature(cddts,trainmask)
            fold_features=pd.concat([fold_features,locales],axis=1)
        if calendar_on:
            fold_features =pd.concat([fold_features, calendar_features],axis=1)
        if relative_on:
            fold_features =pd.concat([fold_features, relative_features],axis=1)
            
        #每次用新模型开始训练
        model= HistGradientBoostingClassifier(**model_parameters)
        model.fit(fold_features.loc[trainmask],labels.loc[trainmask]) #传入测试集以外的集训练
        scores= model.predict_proba(fold_features.loc[validmask]) #对测试集打分
        scores=scores[:,1] #scores返回值为【类别0概率，类别1概率】,这里只取类别1(interested)
        oof.loc[validmask,"score"]= scores

        valid_predictions= oof.loc[validmask]
        user_scores= []
        for user,group in valid_predictions.groupby('user'): #遍历同一user下的所有事件记录
            actual= group.loc[group["interested"]==1, "event"].tolist() #筛选真正感兴趣的事件id
            predicted= group.sort_values(["score",'event'],ascending=[False, True])["event"].tolist() #按score排降序（false），相同时id排升序（true），得到事件推荐顺序

            user_ap= ap(actual,predicted,k)
            user_scores.append(user_ap)

        #每个测试集的map200
        fold_map= float(np.mean(user_scores))
        fold_results.append({"fold":fold_id,"users": len(user_scores),"map200": fold_map,})
        print(f"第{fold_id+1}折 MAP{k}：{fold_map}")
        print()

    return pd.DataFrame(fold_results), oof
