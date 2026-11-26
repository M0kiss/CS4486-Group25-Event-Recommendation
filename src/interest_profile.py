import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

def read_history_pool(cddts, atdeespath):
    '''获取user表示yes/maybe的全部事件'''
    candidate_events={} #一个user相关的全部事件
    histories={} #一个user相关的全部yes/maybe事件

    for user in set(cddts['user']): 
        candidate_events[user]=set()
        histories[user]={
            "yes":set(),
            "maybe":set()
        }

    for user,event in zip(cddts['user'],cddts['event']):
        candidate_events[user].add(event) #上方set自动去重

    with open(atdeespath,mode="r",encoding="utf-8",newline="",) as file:
        reader=csv.DictReader(file)
        for row in reader:
            event=int (row['event'])
            for status in ["yes", "maybe"]:
                people=row[status].split()
                for p in people:
                    user=int(p)

                    if user not in histories:
                        continue
                    if event in candidate_events[user]:
                        continue

                    histories[user][status].add(event) #user所有status的事件

    needed_events=set(cddts['event'])
    for history in histories.values():
        needed_events.update(history["yes"])
        needed_events.update(history["maybe"])

    print("用户yes/maybe池提取结束")
    return histories,needed_events


def read_profile_events(needed_events,eventpath):
    '''根据活动清单查本活动的内容（词频向量）和开始时间'''
    word_columns=[f"c_{i}" for i in range(1,101)]+['c_other']
    columns= ["event_id","start_time"]+word_columns
    reader= pd.read_csv(eventpath,usecols=columns,dtype={"event_id":"int64"},chunksize=100000)

    selected_chunks=[]
    for chunk in reader:
        selected= chunk.loc[chunk["event_id"].isin(needed_events)].copy()
        selected_chunks.append(selected)

    events= pd.concat(selected_chunks,ignore_index=True)
    events= events.set_index("event_id").sort_index()

    start_times = pd.to_datetime(events["start_time"],utc=True,format="mixed",errors="coerce")

    words = events[word_columns].to_numpy(dtype=float)
    words = np.log1p(words)
    words = normalize(words, norm="l2", axis=1)

    content = pd.DataFrame(
        words,
        index=events.index,
        columns=word_columns,
    )

    print("画像活动内容与开始时间读取结束")
    return content, start_times


def build_profile_features(cddts,histories,profile_content,profile_start_times):
    features = pd.DataFrame(index=cddts.index)
    for status in ['yes','maybe']: #初始假设所有用户都没有有效画像
        features[f"{status}_profile_similarity"] =np.nan
        features[f"{status}_history_log_count"]= 0.0
        features[f"{status}_profile_missing"]= True
    for i in cddts.itertuples():
        candidate_vector= profile_content.loc[i.event].to_numpy(dtype=float) #用活动id取处理后的词频内容向量

        for status in ['yes','maybe']:
            valid_events= []
            for event in sorted(histories[i.user][status]):     #取出每个事件timestamp时已经发生的user yes/maybe列表

                if event not in profile_content.index:
                    continue #没有该历史活动的记录则跳过

                start_time= profile_start_times.loc[event]
                if pd.notna(start_time) and start_time <= i.timestamp:
                    valid_events.append(event) 

            features.at[i.Index,f"{status}_history_log_count"]= np.log1p(len(valid_events)) #写入当前事件对应的单元格
            if not valid_events: continue
            history_vectors= profile_content.loc[valid_events].to_numpy(float) #取出所有历史事件的词频向量
            prototype= history_vectors.sum(axis=0)  #对每个词频求和
            length= np.linalg.norm(prototype)
            if length==0:
                continue
            normalized_prototype= prototype / length
            similarity= np.dot(candidate_vector,normalized_prototype) #计算候选事件与prototype的相似度
            features.at[i.Index, f"{status}_profile_similarity"]= similarity
            features.at[i.Index,f"{status}_profile_missing"]= False

    print("本人yes/maybe画像特征生成结束")
    return features




            


