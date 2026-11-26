import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def time_feature(cddts):
    '''关联时间特征'''
    gaptime= (cddts["start_time"]- cddts["timestamp"]).dt.total_seconds()/ (24*60*60) #时间差转换成float天

    features= pd.DataFrame(index= cddts.index)
    features["time_missing"]= gaptime.isna() #标记时间是否已知
    features["days_to_event"]= gaptime.fillna(0).clip(-60, 60) #将过长的gap限制在+-60天
    features["event_weekday"]= (cddts["start_time"].dt.dayofweek.fillna(-1)) #提取星期
    features["log_days_to_event"]= np.log1p(gaptime.fillna(0).clip(lower=0)) #对数时间权重，非正值都取0算对数

    print("时间特征关联结束")
    return features


def normalize_location(value):
    '''格式化地址'''
    if pd.isna(value):
        return ''
    return ' '.join(str(value).lower().split())


def basic_feature(cddts,weights):
    '''关联剩余基础特征'''
    features= pd.DataFrame(index= cddts.index) 
    #受邀   
    features["invited"] = cddts["invited"]

    #时区
    timezone=pd.to_numeric(cddts["timezone"],errors="coerce")
    features['timezone_hours']= timezone.fillna(0)/60 #保存当前时区与utc的偏差值（小时）
    features["timezone_missing"] = cddts["timezone"].isna() #缺失时区

    #坐标缺失?
    features["latitude_missing"]= (cddts["lat"].isna() | cddts["lat"].eq(""))
    features["longitude_missing"]= (cddts["lng"].isna() | cddts["lng"].eq(""))

    #地址匹配度                                                                                                                               可以尝试细化
    user_location= cddts["location"].map(normalize_location) #格式化地址
    location_score= pd.Series(0.0,index=cddts.index)

    for c,w in [("country",weights[0]),("state",weights[1]),("city",weights[2])]:                                                                                      #尝试修改权重
        event_location = cddts[c].map(normalize_location)

        matches=[]
        for eventtext,usertext in zip(event_location,user_location):
            if eventtext and f" {eventtext} " in f" {usertext} ":  #有事件地点且地点存在用户地址中
                matches.append(True)
            else:
                matches.append(False)
        matched=pd.Series(matches,index=cddts.index)
        location_score= location_score+matched *w
    features["location_score"] = location_score

    print("基础属性特征提取结束")
    return features
            

def content_feature(cddts):
    '''提取事件的词频向量特征 并处理（压缩，归一化）'''
    word_columns= [f"c_{i}" for i in range(1,101)]+["c_other"] #对应事件属性的101词频

    words= cddts[word_columns].to_numpy(dtype=float) #提取每个user-事件组合的101词频矩阵
    words= np.log1p(words) #对数压缩过大词频，降低单纯次数增加带来的影响
    words= normalize(words, norm="l2") #归一化

    print("词频向量提取结束")
    return pd.DataFrame(words,index =cddts.index,columns=word_columns)


def cluster_feature(cddts,content,trainmask,kmeans_parameters):
    '''将训练集事件按类划分'''
    indexs=cddts.loc[trainmask].sort_values("event").drop_duplicates("event").index #取出训练集内每个事件对应的索引（去重

    cluster= KMeans(**kmeans_parameters) #事件分类器
    cluster.fit(content.loc[indexs]) #从当前训练集的词频向量学习类别划分
    cluster_ids = cluster.predict(content) #为所有事件划分类别

    encoded= np.eye(kmeans_parameters['n_clusters'])[cluster_ids]  #将类别转换为独热编码[0,0,0,1,0,0,0,0]表示第4类                                                                        直接用数字记录类别？

    print("事件类别划分结束")
    return pd.DataFrame(encoded,index=cddts.index,columns=[f"cluster_{i}" for i in range(kmeans_parameters['n_clusters'])])


def locale_feature(cddts,trainmask):
    '''locale：用户属性 eg.语言地区设置'''
    locale= cddts['locale'].fillna("missing").astype(str)

    categories= sorted(locale.loc[trainmask].unique()) #读取所有的locale

    features= pd.DataFrame(index=cddts.index)
    for c in categories:
        column_name= f"locale_{c}"
        features[column_name]= locale.eq(c) #逐记录判断是否属于此类别

    print(f"locale提取结束：{len(categories)}列")
    return features


def popular_feature(cddts,atdeespath):
    '''事件参与人数（除自己'''
    needed_event= set(cddts['event'])
    reader= pd.read_csv(atdeespath,usecols=["event", "yes"],dtype={"event": "int64"},chunksize=100_000) #yes试一个事件中已经参与的用户id
    
    event_yes={} #记录每个事件的yes名单
    for chunk in reader:
        selected= chunk.loc[chunk["event"].isin(needed_event)].copy()
        selected["yes"]= selected["yes"].fillna("") #没有参与人标空

        for row in selected.itertuples(index=False):
            people= set(row.yes.split()) #返回事件的yesid名单
            event_yes[row.event]= people

    #计算参与人数
    counts=[]
    for user,event in zip(cddts['user'],cddts['event']):
        people = event_yes[event]
        cnt=len(people)
        if str(user) in people:
            cnt-=1
        counts.append(cnt)
    features = pd.DataFrame(index=cddts.index)
    features["log_total_yes"] = np.log1p(counts) #取对数压缩过大规模差异

    #额外返回raw_yes_cnt用于计算yes/no份额
    raw_yes_cnts = pd.Series(
        counts,
        index=cddts.index,
        dtype=float,
    )

    print("事件yes人数特征提取结束")
    return features,event_yes, raw_yes_cnts


def friendyes_feature(cddts,friendspath,event_yes):#                                                                                                   尝试调整好友yes与所有人yes的比例？
    '''参加事件的好友人数'''
    needed_users= set(cddts["user"])
    reader = pd.read_csv(friendspath,usecols=["user", "friends"],dtype={"user": "int64"},chunksize=100_000)

    user_friends={}
    for chunk in reader:
            selected= chunk.loc[chunk["user"].isin(needed_users)].copy()
            selected["friends"]= selected["friends"].fillna("") #没有好友 标空
    
            for row in selected.itertuples(index=False):
                friends= set(row.friends.split())
                friends = friends - {str(row.user)} #排除本人
                user_friends[row.user] = friends
    
    #计算好友参与事件人数
    counts=[]
    #返回好友总数 用于计算好友yes的比例
    friend_sizes=[]

    for user,event in zip(cddts['user'],cddts['event']):
        people = event_yes[event]
        friends= user_friends[user]
        counts.append(len(friends & people))
        friend_sizes.append(len(friends))
    features = pd.DataFrame(index=cddts.index)
    features["log_friend_yes"] = np.log1p(counts) #取对数压缩过大规模差异

    raw_friend_cnts = pd.Series(
        counts,
        index=cddts.index,
        dtype=float,
    )

    friend_sizes = pd.Series(
        friend_sizes,
        index=cddts.index,
        dtype=float,
    )

    print("好友yes人数特征提取结束")
    return features,raw_friend_cnts,friend_sizes,user_friends


def response_feature(cddts, attendees_path):
    '''事件maybe/no/invited的人数'''
    wanted_events = set(cddts["event"])
    reader = pd.read_csv(
        attendees_path,
        usecols=["event", "maybe", "no", "invited"],
        dtype={"event": "int64"},
        chunksize=100_000,
    )

    event_groups = {}
    for chunk in reader:
        selected = chunk.loc[
            chunk["event"].isin(wanted_events)
        ].copy()

        selected = selected.fillna("")
        for row in selected.itertuples(index=False):
            event_groups[row.event] = {
                "maybe": set(row.maybe.split()),
                "no": set(row.no.split()),
                "invited": set(row.invited.split()),
            }

    #分别记录三种响应人数
    cnts = {
        "maybe": [],
        "no": [],
        "invited": [],
    }
    for user, event in zip(cddts["user"], cddts["event"]):
        groups = event_groups[event]
        person = str(user)
        for status in cnts:
            people = groups[status]
            count = len(people)
            if person in people:
                count -= 1
            cnts[status].append(count)

    features = pd.DataFrame(index=cddts.index)
    for status, values in cnts.items():
        features[f"log_total_{status}"] = np.log1p(values)

    #用于yes/no份额计算
    raw_cnts = pd.DataFrame(
        cnts,
        index=cddts.index,
        dtype=float,
    )

    print("maybe/no/invited人数提取结束")
    return features,raw_cnts,event_groups


def response_ratio(raw_yes_cnts,raw_cnts):
    '''yes/no的份额'''
    denominator=(raw_yes_cnts+raw_cnts["maybe"]+raw_cnts["no"])
    zero_response= denominator.eq(0) #记录maybe/no/yes均没有的事件
    denominator= denominator.replace(0, 1) #把无响应事件人数+1避免0除

    features= pd.DataFrame(index=raw_cnts.index)
    features['yes_share']=raw_yes_cnts/denominator
    features['no_share']=raw_cnts["no"]/denominator
    features["zero_response"] = zero_response

    print('yes/no份额计算结束')
    return features


def friend_ratio(raw_friend_cnts,friend_sizes):
    '''好友yes比例'''
    zero_friend=friend_sizes.eq(0)
    friend_sizes=friend_sizes.replace(0,1)
    
    features= pd.DataFrame(index=friend_sizes.index)
    features["friend_yes_fraction"]= raw_friend_cnts/friend_sizes
    features["zero_friend"] = zero_friend

    print('好友yes比例计算结束')
    return features


def friend_response(cddts,user_friends,event_groups):
    '''好友maybe/no/invited人数'''
    cnts={
        "maybe": [],
        "no": [],
        "invited": [],
    }

    for user,event in zip(cddts["user"],cddts["event"]):
        friends=user_friends[user]
        groups= event_groups[event]

        for status in cnts:
            people= groups[status]
            friend_people= friends & people
            cnts[status].append(len(friend_people))

    #原始人数
    raw_cnts= pd.DataFrame(cnts,index=cddts.index,dtype=float)
    features = pd.DataFrame(index=cddts.index)
    #对数压缩
    for status in cnts:
        features[f"log_friend_{status}"] = np.log1p(raw_cnts[status])

    print("好友maybe/no/invited人数提取结束")
    return features, raw_cnts
    

def friend_response_ratio(raw_counts, friend_sizes):
    '''好友maybe/no/invited比例'''
    friend_sizes = friend_sizes.replace(0, 1)
    features = pd.DataFrame(index=raw_counts.index)
    for status in ["maybe", "no", "invited"]:
        features[f"friend_{status}_fraction"] = (
            raw_counts[status] / friend_sizes
        )

    print("好友maybe/no/invited比例提取结束")
    return features


def original_ratio_feature(raw_yes_counts,raw_counts,raw_friend_counts,raw_other_friend_counts):
    '''5列特征 3总体+2好友'''
    #分别准备总体和好友的yes分母
    total_denominator = raw_yes_counts + 1
    friend_denominator = raw_friend_counts + 1

    features = pd.DataFrame(index=raw_counts.index)

    #总体响应相对于yes的比例
    for status in ["no", "maybe", "invited"]:
        features[f"total_{status}_per_yes"] = (
            raw_counts[status] / total_denominator
        )

    #好友响应相对于好友yes的比例
    for status in ["maybe", "invited"]:
        features[f"friend_{status}_per_yes"] = (
            raw_other_friend_counts[status] / friend_denominator
        )

    print("5列响应比例提取结束")
    return features


def calendar_feature(cddts):
    '''时间特征:活动开始月，小时； 展示月，小时，星期'''
    features = pd.DataFrame(index=cddts.index)

    features["event_month"] = cddts["start_time"].dt.month
    features["event_hour"] = cddts["start_time"].dt.hour

    features["exposure_month"] = cddts["timestamp"].dt.month
    features["exposure_hour"] = cddts["timestamp"].dt.hour
    features["exposure_weekday"] = cddts["timestamp"].dt.dayofweek

    return features


def query_relative_feature(cddts,features):
    '''同时间批次事件中的相对排名特征'''
    groupers=[cddts["user"],cddts['timestamp']] #找出一个user在同一批次被展示的活动列表

    result= pd.DataFrame(index=cddts.index)
    columns=["log_total_yes","log_friend_yes","yes_share","friend_yes_fraction"]

    for c in columns:
        result[f"{c}_batch_percentile"]= features[c].groupby(groupers).rank(method='average', pct= True) #对同user同时段内的所有候选事件的c属性进行排名， pct=true 再除以本组数量得到归一化相对排名

    proximity= -features['days_to_event'].abs() #展示时间离开始时间越近，值越大
    result['time_proximity_batch_percentile']= proximity.groupby(groupers).rank(method='average',pct=True)

    batch_size= cddts['user'].groupby(groupers).transform('size') #统计每批的事件数量
    result["log_batch_size"] = np.log1p(batch_size) #对数压缩

    print("同时段事件特征提取结束")
    return result
