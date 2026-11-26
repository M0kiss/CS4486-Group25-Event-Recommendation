from pathlib import Path
import pandas as pd
from feature import time_feature,basic_feature,content_feature,popular_feature,friendyes_feature,response_feature,   response_ratio,friend_ratio,friend_response,friend_response_ratio,original_ratio_feature
from validation import make_fold,ap
from model import run_cv
from interest_profile import read_history_pool,read_profile_events,build_profile_features
from geography import read_place_lookup,advanced_geo_feature

PROJECT_ROOT= Path(__file__).resolve().parent.parent
RAW_DIR= PROJECT_ROOT / "data" / "raw"
trainpath= RAW_DIR / "train.csv"      #训练集path
userpath= RAW_DIR / "users.csv"
eventpath= RAW_DIR / "events.csv"
atdeespath= RAW_DIR / "event_attendees.csv"
friendspath= RAW_DIR / "user_friends.csv"


#参数                                                                                                                             (你们可以调这里试试) 现在的配置是我试出来最好的
seed             = 25                #划分user fold的随机种子
foldnum          = 5                 #折数
apnum            = 200               #筛选用于打分的前apnum个候选事件
location_weights = [1,1,2]           #country,state,city的权重

#模型参数
max_iter         = 300	             #进行x轮提升
max_leaf_nodes   = 31	             #每棵树最多x个叶节点
min_samples_leaf = 60	             #每个叶节点至少包含x个训练样本
learning_rate    = 0.05	             #控制每轮模型更新的幅度
l2_regularization= 10	             #对叶节点数值施加正则约束
early_stopping   = False	         #不自动提前结束
random_state     = seed              #模型随机种子，这里服用seed                   

#kmeans参数 用于划分事件类别
cluster_on       = True              #是否启用 +
n_clusters       = 20                #将所有事件划分为x类
n_init           = 10                #共尝试x组分类，择优
max_iter_kmeans  = 300               #最多x轮
random_state     = seed              #复用seed

#额外特征                                                                                                                        （或者加一些额外特征尝试一下）
locale_on        = True              #是否加入locale特征
popular_on       = True              #事件参与人数
friendyes_on     = True              #事件中的好友参与人数 
response_on      = True              #事件里表示可能参加/不参加/被邀请的人数
yesnoratio_on    = True              #yes/no的整体比例
friendratio_on   = True              #事件在好友里的yes比例
friendresponse_on= True              #好友可能参加/不参加/被邀请的人数
frndrspsratio_on = True              #好友可能参加/不参加/被邀请的比例
profile_on       = True              #用户历史画像
originalratio_on = True              #5列响应比例 eg.总体no/(总体yes+1)
advancedgeo_on   = False             #进阶地理匹配 （这个目前是负作用，所以关了）
calendar_on      = True              #时间特征x5
relative_on      = True              #同时间批次下事件的相对排名特征

def load_candidate(trainpath):
    '''返回处理后的训练记录'''
    train= pd.read_csv(trainpath,dtype={"user":"int64","event":"int64"})
    print("load_cadidate开始： ", len(train)," 行candidates")

    required_columns={"user","event","invited","timestamp","interested","not_interested"}       #必须字段
    if required_columns - set(train.columns):
        raise ValueError(f"训练集缺少字段：{required_columns - set(train.columns)}")

    #时间格式化
    train['timestamp']=pd.to_datetime(train['timestamp'],utc=True,format='mixed',errors='raise')
    if train["timestamp"].isna().any():
        raise ValueError("时间缺失")
    
    train= train.sort_values("timestamp") #按时间排序
    train= train.drop_duplicates(      #删去重复记录   
        subset=["user", "event"],
        keep="first",
    )
    train = train.reset_index(drop=True)

    print("load_cadidate结束： ", len(train)," 行candidates")
    return train


def read_userfeatures(train_cddts, userpath):
    '''将user features关联到user-事件组合'''
    users= pd.read_csv(userpath,usecols=["user_id","locale","location","timezone"], dtype={"user_id":"int64"}) #只读取usecols的属性
    result=(train_cddts.merge(users,how='left',left_on="user",right_on="user_id")).drop(columns=['user_id']) #合并特征-组合,去掉重复的id列
    print(f"用户特征关联结束：{len(result)} 行")
    return result


def read_eventfeatures(cddts, eventpath):
    '''将event features关联到train'''
    needed_event= set(cddts["event"])
    required_columns=['event_id','start_time','city','state','country','lat','lng'] #lat，lng经纬度
    word_columns= [f"c_{i}" for i in range(1,101)]+["c_other"] #对应事件属性的101维特征
    required_columns+=word_columns
    reader=pd.read_csv(eventpath,usecols=required_columns,dtype={"event_id":"int64"},chunksize=100000) #分批读取

    selected=[]
    for i in reader:
        selecting= i.loc[i["event_id"].isin(needed_event)].copy() #只复制需要的事件
        selected.append(selecting)
    events= pd.concat(selected,ignore_index=True)

    events["start_time"] = pd.to_datetime(events["start_time"],utc=True,format="mixed",errors="coerce") #时间格式化

    result=(cddts.merge(events,how='left',left_on='event',right_on='event_id')).drop(columns=['event_id'])
    print("事件特征关联结束")
    return result


if __name__=="__main__":
    train= load_candidate(trainpath)
    print(train.head())
    print() 

    train= read_userfeatures(train,userpath)
    display_columns= [
        "user",
        "event",
        "interested",
        "locale",
        "location",
        "timezone",
    ]
    print(train[display_columns].head())
    print()

    train= read_eventfeatures(train,eventpath)
    display_columns = [
        "user",
        "event",
        "timestamp",
        "start_time",
        "location",
        "city",
        "country",
        "interested",
    ]
    print(train[display_columns].head())
    print() 

    time_features= time_feature(train)
    print(time_features.head())
    print() 

    basic_features= basic_feature(train,location_weights)
    features= pd.concat([time_features,basic_features],axis=1) #横向拼接
    feature_columns = [
        "invited",
        "days_to_event",
        "time_missing",
        "location_score",
        "log_days_to_event",
        "timezone_hours",
        "timezone_missing",
        "latitude_missing",
        "longitude_missing",
        "event_weekday",
    ]
    features = features[feature_columns]
    print(features.head())
    print() 

    # 总体yes、好友yes、响应比例都需要yes名单或人数
    if popular_on or friendyes_on or yesnoratio_on or friendratio_on or friendresponse_on or frndrspsratio_on or originalratio_on:
        popularity, event_yes, raw_yes_counts = popular_feature(
            train,
            atdeespath,
        )

        if popular_on:
            features = pd.concat(
                [features, popularity],
                axis=1,
            )

        if friendyes_on or friendratio_on or friendresponse_on or frndrspsratio_on or originalratio_on:
            (
                friend_features,
                raw_friend_counts,
                friend_sizes,
                user_friends,
            ) = friendyes_feature(
                train,
                friendspath,
                event_yes,
            )

            # 可选：好友yes人数
            if friendyes_on:
                features = pd.concat(
                    [features, friend_features],
                    axis=1,
                )

            # 可选：好友yes比例及零好友标记
            if friendratio_on:
                friend_ratios = friend_ratio(
                    raw_friend_counts,
                    friend_sizes,
                )

                features = pd.concat(
                    [features, friend_ratios],
                    axis=1,
                )

    # 其他响应人数及比例需要maybe/no/invited数据
    if response_on or yesnoratio_on or friendresponse_on or frndrspsratio_on or originalratio_on:
        response_features, raw_counts, event_groups = (
            response_feature(
                train,
                atdeespath,
            )
        )

        if response_on:
            features = pd.concat(
                [features, response_features],
                axis=1,
            )

    if yesnoratio_on:
        response_ratios = response_ratio(
            raw_yes_counts,
            raw_counts,
        )

        features = pd.concat(
            [features, response_ratios],
            axis=1,
        )

    # 人数或比例任一开启，都需要先计算原始人数
    if friendresponse_on or frndrspsratio_on or originalratio_on:
        friend_responses, raw_other_friend_counts = (
            friend_response(
                train,
                user_friends,
                event_groups,
            )
        )

        # 可选：加入对数人数
        if friendresponse_on:
            features = pd.concat(
                [features, friend_responses],
                axis=1,
            )

        # 可选：加入好友比例
        if frndrspsratio_on:
            friend_response_ratios = friend_response_ratio(
                raw_other_friend_counts,
                friend_sizes,
            )

            features = pd.concat(
                [features, friend_response_ratios],
                axis=1,
            )

    if originalratio_on:
        original_ratios = original_ratio_feature(
            raw_yes_counts,
            raw_counts,
            raw_friend_counts,
            raw_other_friend_counts,
        )

        features = pd.concat(
            [features, original_ratios],
            axis=1,
        )

        print(original_ratios.head())

    if profile_on:
        histories, profile_events = read_history_pool(
            train,
            atdeespath
        )

        profile_content, profile_start_times = read_profile_events(
            profile_events,
            eventpath
        )

        profile_features = build_profile_features(
            train,
            histories,
            profile_content,
            profile_start_times
        )

        print(profile_features.head())

        features = pd.concat(
            [features, profile_features],
            axis=1
        )

    if advancedgeo_on:
        countries, city_to_places = read_place_lookup(eventpath)

        geo_features = advanced_geo_feature(
            train,
            countries,
            city_to_places,
        )

        features = pd.concat(
            [features, geo_features],
            axis=1,
        )

        print(geo_features.head())
        
    folds=make_fold(train,foldnum,seed)
    print("每折用户数：\n",
        train["user"]
        .groupby(folds)
        .nunique()
        .sort_index()
    )
    print() 

    model_parameters = {
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "min_samples_leaf": min_samples_leaf,
        "learning_rate": learning_rate,
        "l2_regularization": l2_regularization,
        "early_stopping": early_stopping,
        "random_state": random_state,
    }
    kmeans_parameters = {
        "n_clusters": n_clusters,
        "n_init": n_init,
        "max_iter": max_iter_kmeans,     
        "random_state": random_state,
    }
    content = None
    if cluster_on:
        content = content_feature(train)
    fold_results,oof= run_cv(train,features,folds,apnum,model_parameters,kmeans_parameters,content,locale_on,calendar_on,relative_on)
    mean_map = fold_results["map200"].mean()
    print(f"五折平均 MAP{apnum}：{mean_map}")
    print()



    
