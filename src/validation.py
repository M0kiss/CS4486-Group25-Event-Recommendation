import numpy as np
import pandas as pd

def make_fold(cddts,num,seed):
    '''将用户随机num折'''
    users=np.sort(cddts["user"].unique()) #不重复user
    random=np.random.default_rng(seed)
    random.shuffle(users)

    user_mapping={}
    for i, user in enumerate(users):
        user_mapping[user]=i%num
    folds = cddts["user"].map(user_mapping)

    print(f'用户分组完成: {num}组，种子{seed}')
    return folds


def ap(actual, predicted, k):
    '''返回一个用户的ap200'''
    actual= set(actual)
    if not actual:
        return 0.0

    hit= 0
    score= 0.0
    seen= set() #已经出现过的事件集合

    for rank, event in enumerate(predicted[:k],start=1):
        if event in actual and event not in seen:
            hit+=1
            score+=hit/rank     #到此为止时 中的/总数

        seen.add(event)

    return score/min(len(actual),k) #返回归一化后的ap分数


    