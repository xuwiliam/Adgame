import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score

import BayesianSmooth
import types
import FeatureBuilder
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec


one_hot_feature = ['LBS', 'age', 'carrier', 'consumptionAbility', 'education', 'gender', 'house', 'os',
                   'marriageStatus', 'advertiserId', 'campaignId', 'creativeId', 'adCategoryId', 'productId',
                   'productType']
vectorFeature = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3',
                 'topic1', 'topic2', 'topic3']
ratio_feature = ['advertiserId','campaignId','creativeId','creativeSize','adCategoryId','productId','productType',
             'LBS','age','appIdAction','appIdInstall','carrier','consumptionAbility','ct','education','gender','house','interest1',
             'interest2','interest3','interest4','interest5','kw1','kw2','kw3','marriageStatus','os','topic1','topic2','topic3']

def CountId(path1, path2):
    f1 = open(path1)
    f2 = open(path2)
    train_uid=set()
    test_uid=set()
    train_aid=set()
    test_aid=set()
    train_pair=set()
    test_pair=set()
    for line in f1.readlines():
        data = line[:-1].split(',')
        tuple = [data[0],data[1]]
        train_pair.add(tuple)
        train_uid.add(data[1])
        train_aid.add(data[0])
    for line in f2.readlines():
        data = line.split(',')
        tuple = [data[0],data[1]]
        test_uid.add(data[1])
        test_aid.add(data[0])
        test_pair.add(tuple)
    print(ListUnion(train_pair,test_pair))
    print(len(ListUnion(train_pair,test_pair)))
    print('len train_pair: ',len(train_pair),'len test_pari: ',len(test_pair))
    print(ListUnion(train_uid,test_uid))
    print(len(ListUnion(train_uid,test_uid)))
    print('len train_uid: ',len(train_uid),'len test_uid',len(test_uid))
    print(ListUnion(train_aid,test_aid))
    print(len(ListUnion(train_aid,test_aid)))
    print('len train_aid:',len(train_aid),'len test_aid: ',len(test_aid))
def ListUnion(collection1,collection2):
    return list(collection1.intersection(collection2))

def StaticAdPositiveAndNegative(path1,path2):
    fp1 = open(path1)
    fp2 = open(path2)
    aid_positive={}
    aid_negative={}
    aid_train={}
    aid_test={}
    for line in fp1.readlines():
        data = line[:-1].split(',')
        if data[1] not in aid_positive:
           aid_positive[data[1]]=0
        if data[2]==1:
              aid_positive[data[1]]+=1
        if data[1] not in aid_negative:
           aid_negative[data[1]]=0
        if data[2]==-1:
           aid_negative[data[1]]+=1
        if data[1] not in aid_train:
           aid_train[data[1]]=0
        aid_train[data[1]]+=1
    for line in fp2.readlines():
        data = line[:-1].split(',')
        if data[1] not in aid_test:
           aid_test[data[1]]=0
        aid_test[data[1]]+=1

    for key in aid_positive:
        print(aid_positive[key],aid_negative[key])
        print('positive: ',aid_positive[key]*1.0/(aid_positive[key]+aid_negative[key]))
        print('negative: ',aid_negative[key]*1.0/(aid_positive[key]+aid_negative[key]))

    for key in aid_train:
        print(key,'train propotion: ',aid_train[key]/8798815.0)
        print(key,'test propotion: ',aid_test[key]/2265990.0)
def InterestTopicKeyWordStatics(path):
    in1=set()
    in2=set()
    in3=set()
    in4=set()
    in5=set()
    topic1=set()
    topic2=set()
    topic3=set()
    keyword1=set()
    keyword2=set()
    keyword3=set()
    fp = open(path)
    for line in fp.readlines():
        data = line.strip().split('|')
        for i in range(7,len(data)):
            arr = data[i].split(' ')
            if arr[0]=='interest1':
               for j in range(1,len(arr)):
                   in1.add(arr[j])
            elif arr[0]=='interest2':
               for j in range(1, len(arr)):
                   in2.add(arr[j])
            elif arr[0] == 'interest3':
                for j in range(1, len(arr)):
                    in3.add(arr[j])
            elif arr[0] == 'interest4':
                for j in range(1, len(arr)):
                    in4.add(arr[j])
            elif arr[0] == 'interest5':
                for j in range(1, len(arr)):
                    in5.add(arr[j])
            elif arr[0] == 'topic1':
                for j in range(1, len(arr)):
                    topic1.add(arr[j])
            elif arr[0] == 'topic2':
                for j in range(1, len(arr)):
                    topic2.add(arr[j])
            elif arr[0] == 'topic3':
                for j in range(1, len(arr)):
                    topic3.add(arr[j])
            elif arr[0] == 'kw1':
                for j in range(1, len(arr)):
                    keyword1.add(arr[j])
            elif arr[0] == 'kw2':
                for j in range(1, len(arr)):
                    keyword2.add(arr[j])
            elif arr[0] == 'kw3':
                for j in range(1, len(arr)):
                    keyword3.add(arr[j])
    print(in1)
    print(in2)
    print(in3)
    print(in4)
    print(in5)
    print(topic1)
    print(topic2)
    print(topic3)
    print(keyword1)
    print(keyword2)
    print(keyword3)
def Sysmetricdiff(path1,path2,data_feature):
    train_interest_1 = set()
    train_interest_2 = set()
    train_interest_3 = set()
    train_interest_4 = set()
    train_interest_5 = set()
    train_kw_1 = set()
    train_kw_2 = set()
    train_kw_3 = set()
    train_topic_1 = set()
    train_topic_2 = set()
    train_topic_3 = set()
    train_app_action = set()
    train_app_install = set()
    test_interest_1 = set()
    test_interest_2 = set()
    test_interest_3 = set()
    test_interest_4 = set()
    test_interest_5 = set()
    test_kw_1 = set()
    test_kw_2 = set()
    test_kw_3 = set()
    test_topic_1 = set()
    test_topic_2 = set()
    test_topic_3 = set()
    test_app_action = set()
    test_app_install = set()
    fp1 = open(path1)
    fp2 = open(path2)
    feature_line=[12,13,20,21,22,23,24,25,26,27,30,31,32]
    for line in fp1.readlines():
        data = line.strip().split(',')
        for i in feature_line:
            arr = data[i].split(' ')
            find_property(arr,train_interest_1,train_interest_2,train_interest_3,train_interest_4,train_interest_5
                          ,train_topic_1,train_topic_2,train_topic_3,train_kw_1,train_kw_2,train_kw_3,train_app_action,
                          train_app_install)
    for line in fp2.readlines():
        data = line.strip().split(',')
        for i in feature_line:
            arr = data[i].split(' ')
            find_property(arr,test_interest_1,test_interest_2,test_interest_3,test_interest_4,test_interest_5,test_topic_1,
                          test_topic_2,test_topic_3,test_kw_1,test_kw_2,test_kw_3,test_app_action,test_app_install)

    print('interest1_symmetric: ', train_interest_1.symmetric_difference(test_interest_1))
    print('interest2 symmetric: ', train_interest_2.symmetric_difference(test_interest_2))
    print('interest3 symmetric: ', train_interest_3.symmetric_difference(test_interest_3))
    print('interest4 symmetric: ', train_interest_4.symmetric_difference(test_interest_4))
    print('interest5 symmetric: ', train_interest_5.symmetric_difference(test_interest_5))
    print('topic1_symmetric: ',train_topic_1.symmetric_difference(test_topic_1))
    print('topic2_symmetric: ',train_topic_2.symmetric_difference(test_topic_2))
    print('topic3_symmetric: ', train_topic_3.symmetric_difference(test_topic_3))
    print('kw1_symmetric: ',train_kw_1.symmetric_difference(test_kw_1))
    print('kw2_symmetric: ',train_kw_2.symmetric_difference(test_kw_2))
    print('kw3_symmetric: ',train_kw_3.symmetric_difference(test_kw_3))



def find_property(data,interest_1,interest_2,interest_3,interest_4,interest_5,
                  topic1,topic2,topic3,kw1,kw2,kw3,app_action=None,app_install=None):
    l = len(data)
    if data[0] == 'interest1':
        for j in range(1, l):
            interest_1.add(data[j])
    elif data[0] == 'interest2':
        for j in range(1, l):
            interest_2.add(data[j])
    elif data[0] == 'interest3':
        for j in range(1, l):
            interest_3.add(data[j])
    elif data[0] == 'interest4':
        for j in range(1, l):
            interest_4.add(data[j])
    elif data[0] == 'interest5':
        for j in range(1, l):
            interest_5.add(data[j])
    elif data[0] == 'topic1':
        for j in range(1, l):
            topic1.add(data[j])
    elif data[0] == 'topic2':
        for j in range(1, l):
            topic2.add(data[j])
    elif data[0] == 'topic3':
        for j in range(1, l):
            topic3.add(data[j])
    elif data[0] == 'kw1':
        for j in range(1, l):
            kw1.add(data[j])
    elif data[0] == 'kw2':
        for j in range(1, l):
            kw2.add(data[j])
    elif data[0] == 'kw3':
        for j in range(1, l):
            kw3.add(data[j])
    if app_action != None and data[0]=='appIdInstall':
       app_action.add(data[1])
    if app_install != None and data[0]=='appIdAction':
       app_install.add(data[1])
def Interest_kw_topic():
    train_sample = ReadCombineTrain()
    dict_interest = {}
    dict_negative = {}
    negative=set()
    positive=set()
    for data in train_sample:
        if data[2]=='0':
           if data[0] not in dict_interest:
              dict_interest[data[0]]=set()

           items = data[20].split(' ')
           l = len(items)
           for i in range(l):
               dict_interest[data[0]].add(items[i])
               positive.add(items[i])
        if data[2]=='1':
           if data[0] not in dict_negative:
              dict_negative[data[0]] = set()
           items = data[20].split(' ')
           l = len(items)
           for i in range(l):
               dict_negative[data[0]].add(items[i])
               negative.add(items[i])
    pre = None
    dict_value_propotion_rank = {}
    dict_value_propotion_negative_rank = {}
    sum = 0
    for key in dict_interest:
        if pre == None:
           pre = dict_interest[key]
        else:
           pre = dict_interest[key].intersection(pre)
           print('pre',pre)
           if len(pre) == 0:
              pre = None
              print('None')
        for value in dict_interest[key]:
            if value not in dict_value_propotion_rank:
               dict_value_propotion_rank[value]=0
            dict_value_propotion_rank[value]+=1
            sum+=1
    for key in dict_value_propotion_rank:
        dict_value_propotion_rank[key]/=(sum*1.0)

    res=sorted(dict_value_propotion_rank.items(),key=lambda item:item[1])
    print(dict_value_propotion_rank)

def ReadCombineTrain():
    fp = open('/home/niezhaochang/ad-game/data/combine_train.csv','r')
    sample = []
    for line in fp.readlines():
        data = line.split(',')
        sample.append(data)
    return sample
def repeatUid():
    fp = open('/home/niezhaochang/ad-game/data/test.csv','r')
    repeatdict={}
    for line in fp.readlines():
        data = line.strip().split(',')
        if data[1] not in repeatdict:
           repeatdict[data[1]]=[]
        repeatdict[data[1]].append(data[2])
    repeat_positive = {}
    for key in repeatdict:
        if len(repeatdict[key])>1:
           for i in range(0,len(repeatdict[key])):
               if repeatdict[key][i] == '1':
                  if key not in repeat_positive:
                     repeat_positive[key]=0
                  repeat_positive[key]+=1

    print('len_repeat_positive: ',len(repeat_positive))
    sum = 0
    for key in repeat_positive:
        sum+=repeat_positive[key]

        print(key,repeat_positive[key])
    print(sum-len(repeat_positive))

def XGBoost():
    train_sample = ReadCombineTrain()

    clf = xgb()
def StaticsRepeatkwTopic():
    train_sample = ReadCombineTrain()
    feature_dict={}
    feature_dict_positive={}
    keyword1_uid=set()
    keyword2_uid=set()
    keyword3_uid=set()
    topic1_uid = set()
    topic2_uid = set()
    topic3_uid = set()
    feature=['kw1','kw2','kw3','topic1','topic2','topic3']
    for data in train_sample:

        if data[2]=='1':

            if data[25] != '-1':
                if feature[0] not in feature_dict_positive:
                    feature_dict_positive[feature[0]] = 0
                feature_dict_positive[feature[0]] += 1
                keyword1_uid.add(data[0])
            if data[26] != '-1':
                if feature[1] not in feature_dict_positive:
                    feature_dict_positive[feature[1]] = 0
                feature_dict_positive[feature[1]] += 1
                keyword2_uid.add(data[0])
            if data[27] != '-1':
                if feature[2] not in feature_dict_positive:
                    feature_dict_positive[feature[2] ] = 0
                feature_dict_positive[feature[2] ] += 1
                keyword3_uid.add(data[0])
            if data[30] != '-1':
                if feature[3]  not in feature_dict_positive:
                    feature_dict_positive[feature[3]] = 0
                feature_dict_positive[feature[3]] += 1
                topic1_uid.add(data[0])
            if data[31] != '-1':
                if feature[4] not in feature_dict_positive:
                    feature_dict_positive[feature[4]] = 0
                feature_dict_positive[feature[4]] += 1
                topic2_uid.add(data[0])
            if data[32] != '-1':
                if feature[5] not in feature_dict_positive:
                    feature_dict_positive[feature[5]] = 0
                feature_dict_positive[feature[5]] += 1
                topic3_uid.add(data[0])
        else:
            if data[25] != '-1':
                if feature[0] not in feature_dict:
                    feature_dict[feature[0]] = 0
                feature_dict[feature[0]] += 1
            if data[26] != '-1':
                if feature[1] not in feature_dict:
                    feature_dict[feature[1]] = 0
                feature_dict[feature[1]] += 1

            if data[27] != '-1':
                if feature[2] not in feature_dict:
                    feature_dict[feature[2]] = 0
                feature_dict[feature[2]] += 1
            if data[30] != '-1':
                if feature[3] not in feature_dict:
                    feature_dict[feature[3]] = 0
                feature_dict[feature[3]] += 1
            if data[31] != '-1':
                if feature[4] not in feature_dict:
                    feature_dict[feature[4]] = 0
                feature_dict[feature[4]] += 1
            if data[32] != '-1':
                if feature[5] not in feature_dict:
                    feature_dict[feature[5]] = 0
                feature_dict[feature[5]] += 1
    for key in feature_dict:
        print(key,feature_dict[key],feature_dict_positive[key],len(keyword1_uid))

def AppInstall_AppAction():
    train_data = ReadCombineTrain()
    dict_install={}
    dict_action={}
    for data in train_data:
       act_val = data[12].split(' ')
       install_val = data[13].split(' ')
       for val in act_val:
         if val not in dict_action:
            dict_action[val]={}
         if data[2] not in dict_action[val]:
               dict_action[val][data[2]]=0
         dict_action[val][data[2]]+=1

       for val in install_val:
         if val not in dict_install:
            dict_install[val]={}
         if data[2] not in dict_install[val]:
              dict_install[val][data[2]]=0
         dict_install[val][data[2]]+=1
    sum_action=0
    sum_action_positive=0
    for key in dict_action:
           sum_action+=dict_action[key]['0']
           sum_action+=dict_action[key]['1']
           sum_action_positive+=dict_action[key]['1']

    sum_install=0
    sum_install_positive=0
    for key in dict_install:
           sum_install+=dict_install[key]['0']
           sum_install+=dict_install[key]['1']
           sum_install_positive+=dict_install[key]['1']

    print('app_action value != -1: ',(sum_action_positive-dict_action['-1']['1'])*1.0/sum_action)
    print('app_action value == -1: ',(dict_action['-1']['1']*1.0)/sum_action)
    print('app_install value !=  -1: ',(sum_install_positive-dict_install['-1']['1'])*1.0/sum_install)
    print('app_install value == -1: ',dict_install['-1']['1']*1.0/sum_install)

    print('appinstall value = 1: ',dict_install['1']['1'],(dict_install['1']['1']*1.0)/(dict_install['1']['1']
                                                                                  +dict_install['1']['0']))
    print('appinstall value = -1: ',dict_action['1'])

def StaticsFeatureRatio(gbdt_train):
    feature=['aid','uid','label','advertiserId','campaignId','creativeId','creativeSize','adCategoryId','productId','productType',
             'LBS','age','appIdAction','appIdInstall','carrier','consumptionAbility','ct','education','gender','house','interest1',
             'interest2','interest3','interest4','interest5','kw1','kw2','kw3','marriageStatus','os','topic1','topic2','topic3']

    key_value_ratio={}
    for i,key in enumerate(feature):
        if i>2:
           key_value_ratio[key]={}

        values = gbdt_train[key].values
        label  = gbdt_train[key].values
        for i,v in enumerate(values):
            if v!=-1 and label[i]!=-1:
               if isinstance(v,int):
                   if v not in key_value_ratio[key]:
                       key_value_ratio[key][v]['all']=0
                       key_value_ratio[key][v]['1']=0
                   if label[i] == 1:
                      key_value_ratio[key][v]['1']+=1
                   key_value_ratio[key][v]['all']+=1
               if isinstance(v,list):
                   v_detail = v.split(' ')
                   for i,d in enumerate(v_detail):
                     if d not in key_value_ratio[key]:
                        key_value_ratio[key][d]={}
                        key_value_ratio[key][d]['all']=0
                     if label[i]==1:
                       if label[i] not in key_value_ratio[key][d]:
                           key_value_ratio[key][d]['1']=0
                       key_value_ratio[key][d]['1']+=1
                     key_value_ratio[key][d]['all']+=1
    for key in key_value_ratio:
        f = open(key+'_count','w+')
        for d in key_value_ratio[key]:
            all = key_value_ratio[key][d]['all']
            positive = key_value_ratio[key][d]['1']
            hyper = BayesianSmooth.HyperParam(1,1)
            alpha,beta = hyper.update_from_data_by_FPI(all,positive,1000,0.00000001)
            ratio = (alpha+positive)*1.0/((beta+all)*1.0)
            f.write(str(d)+','+str(ratio))
def StaticUser(train_user,test_user):
    train_dict = {}
    f1 = open('train_user_frequence.txt','w+')
    f2 = open('test_user_frequence.txt','w+')
    test_dict = {}
    for user in train_user['uid'].values:
        if user not in train_dict:
           train_dict[user] = 0
        train_dict[user]+=1
    for user in test_dict['uid'].values:
        if user not in test_dict:
           test_dict[user]=0
        test_dict[user]+=1
    for key in train_dict:
        f1.write(key,str(train_dict[key])+'\n')
    f1.close()
    for key in test_dict:
        f2.write(key,str(test_dict[key])+'\n')
    f2.close()
def InsertFeature(train_data,test_data,insertfeature,feature_key):
    train_ = []
    test_ = []
    for i, item in enumerate(train_data):
        data = str(item).split(' ')
        sum_ = 0
        for d in data:

            if d == '-1':
                continue
            key = str(int(float(d)))
            if key not in insertfeature:
                continue
            sum_ += insertfeature[key]
        train_.append(sum_)
    for i, item in enumerate(test_data):
        data = str(item).split(' ')
        sum_ = 0
        for d in data:
            if d == '-1':
                continue
            key = str(int(float(d)))
            if key not in insertfeature:
                continue
            sum_ += insertfeature[key]
        test_.append(sum_)
    return train_, test_


def MergeFeature(combine_train,combine_test):
    x_train=combine_train[['uid']]
    y_train=combine_train[['label']]
    x_tmp = LabelEncoder.fit_transform(combine_train['uid'].values.reshape(-1,1))
    aid = LabelEncoder.fit_transform(combine_train['aid'].values.reshape(-1,1))
    x_train = sparse.hstack((x_tmp,aid))
    y_train = combine_train['label'].values.reshape(-1,1)
    x_verify = LabelEncoder.fit_transform(combine_test['uid'].values.reshape(-1,1))
    aid = LabelEncoder.fit_transform(combine_train['aid'].values.reshape(-1,1))
    x_verify = sparse.hstack((x_verify,aid))

    cv = CountVectorizer()
    for feature in one_hot_feature:
        train_tmp = OneHotEncoder.fit_transform(combine_train[feature].values.reshape(-1,1))
        x_train = sparse.hstack((x_train,train_tmp))
        test_tmp = OneHotEncoder.fit_transform(combine_test[feature].values.reshape(-1,1))
        x_verify = sparse.hstack((x_verify,test_tmp))
    for feature in vectorFeature:
        train_tmp = cv.fit_transform(combine_train[feature].values.reshape(-1,1))
        x_train = sparse.hstack((x_train,train_tmp))
        test_tmp = cv.fit_transform(combine_test[feature].values.reshape(-1,1))
        x_verify = sparse.hstack((x_verify,test_tmp))
    cnt = 0
    df = [0] * 31
    df2 = [0] * 31

    for f in ratio_feature:
        df[cnt] = pd.read_csv(f + '_count', sep=',', header=None)
        df2[cnt] = pd.DataFrame(df[cnt], columns=['value', 'ratio'])
        dict_ = {}
        for row in df2[cnt].iterrows():
            dict_[row['value']] = row['ratio']
            train_res, test_res = InsertFeature(combine_train[f].values, combine_test[f].values, dict_, f)
            x_train = sparse.hstack((x_train,np.array(train_res).reshape(-1,1)))
            x_verify = sparse.hstack((x_verify,np.array(test_res).reshape(-1,1)))
    return x_train,x_verify,y_train
def CombineAdUserFeatureStatic(train_data):
    adFeature = ['aid','advertiserId','campaignId','creativeId','creativeSize','adCategoryId','productId','productType']
    userFeature = ['LBS','age','appIdAction','appIdInstall','carrier','consumptionAbility','ct','education',
                   'gender','house','interest1','interest2','interest3','interest4','interest5',
                   'kw1','kw2','kw3','marriageStatus','os','topic1','topic2','topic3']
    labels = train_data['label'].values
    f_save = [0]*(8*23+1)
    cnt = 0
    for f1 in adFeature:

        for f2 in userFeature:
            dict_pos = {}
            dict_all = {}
            bayesian_pos=0
            data1 = train_data[f1].values
            data2 = train_data[f2].values
            bayesian_all = len(data1)
            for i,item in enumerate(data1):
                d1 = str(item).split(' ')
                d2 = str(data2[i]).split(' ')
                if str(labels[i])=='1':
                  bayesian_pos+=1
                for x1 in d1:
                    for x2 in d2:
                        key = x1+'|'+x2
                        if key not in dict_pos:
                           dict_pos[key]=0
                        if key not in dict_all:
                           dict_all[key]=0
                        dict_all[key]+=1
                        if str(labels[i])=='1':
                           dict_pos[key]+=1
            f_save[cnt] = open(f1+'_'+f2+'_'+'count','w+')
            for key in dict_pos:
                all = dict_all[key]
                positive = dict_pos[key]
                hyper = BayesianSmooth.HyperParam(1, 1)
                I, C = hyper.sample_from_beta(bayesian_pos, bayesian_all, 100,
                                              bayesian_all)
                alpha, beta = hyper.update_from_data_by_FPI(I,C, 100, 0.00000001)
                ratio = (alpha + positive) * 1.0 / ((beta + all) * 1.0)
                f_save[cnt].write(key + ',' + str(ratio))
def runXGB(data_train,data_test):
    train_set = [[]]
    test_set = [[]]
    y_train = data_train['label'].values
    cnt = 0
    df = [0] * 31
    df2 = [0] * 31
    for f in ratio_feature:
        df[cnt] = pd.read_csv(f + '_count_final', sep=',', header=None)
        df2[cnt] = pd.DataFrame(df[cnt], columns=['value', 'ratio'])
        dict_ = {}
        for index,row in df2[cnt].iterrows():
            dict_[str(int(float(row['value'])))] = row['ratio']
            train_res, test_res = InsertFeature(data_train[f].values, data_test[f].values, dict_, f)
            train_set = sparse.hstack((train_set, np.array(train_res).reshape(-1, 1)))
            test_set = sparse.hstack((test_set, np.array(test_res).reshape(-1, 1)))
    return train_set,test_set,y_train
def StaticAid(data_train, data_test):
    train_aid = {}
    test_aid = {}
    labels = data_train['label'].value
    aids = data_train['aid'].values
    for i, aid in enumerate(aids):
        if aid not in train_aid:
           train_aid[aid] = {}
           train_aid[aid]['1'] = 0
           train_aid[aid]['all'] = 0
        train_aid[aid]['all'] += 1
        if str(labels[i]) == '1':
           train_aid[aid]['1'] += 1
    aids = data_test['aid'].values
    for i,aid in enumerate(aids):
         if aid not in test_aid:
               test_aid[aid]={}
               test_aid[aid]['all']=0
         test_aid[aid]['all']+=1
    f = open('train_aid_proba.txt','w')
    f1 = open('test_aid_proba.txt','w')
    for key in train_aid:
        ratio1 = train_aid[key]['1']*1.0/train_aid[key]['all']
        ratio2 = train_aid[key]['all']*1.0/45539701.0
        f.write(str(train_aid[key]['1'])+','+str(train_aid[key]['all'])+',    '+str(ratio1)+','+str(ratio2)+'\n')
    f.close()
    for key in test_aid:
         ratio = test_aid[key]['all']/11729074.0
         f1.write(str(test_aid[key]['all'])+','+str(ratio)+'\n')
    f1.close()
def word2v(data):
    for f in vectorFeature:
        sentences=[]
        for i, item in data[f].values:
            row = []
            data  = str(item).split(' ')
            for d in data:
                row.append(d)
            sentences.append(row)
        model = Word2Vec(sentences,sg=0,size=10,window=5,alpha=0.05,workers=5,min_count=2,hs=0)
        model.save(f+'_w2v_model')
def combinefeature(train_data,test_data):
    fp_train = [0]*102
    fp_test = [0]*102
    cnt = 0
    for i,f in enumerate(vectorFeature):
        for j in range(i+1,11):
            fp_train[cnt] = open(f+'_'+vectorFeature[j]+'_train_combine.txt','w+')
            train_a = train_data[f].values
            train_b = train_data[vectorFeature[j]].values
            for k,item in enumerate(train_a):
                combine_x1=[]
                d = str(item).split(' ')
                b = str(train_b[k]).split(' ')
                for  dd in enumerate(d):
                    for bb in enumerate(b):
                       combine_x1.append(dd+'|'+bb)
                l = 0
                res=''
                for l in range(0,len(combine_x1)-1):
                    res = res + combine_x1[l]+','
                res = res+combine_x1[-1]
                fp_train[cnt].write(res+'\n')
            fp_train[cnt].close()
            train_a=None
            train_b=None
            test_a = test_data[f].values
            test_b = test_data[vectorFeature[j]].values
            for k, item in enumerate(test_a):
                combine_x2 = []
                c = str(item).split(' ')
                f = str(test_b[k]).split(' ')
                for cc in enumerate(c):
                    for ff in enumerate(f):
                        combine_x2.append(cc+'|'+ff)
                z = 0
                res=''
                for z in range(0,len(combine_x2)-1):
                    res = res+combine_x2[z]+','
                res = res+combine_x2[-1]
                fp_test[cnt].write(res+'\n')
            fp_test[cnt].close()
            test_a=None
            test_b=None
            cnt+=1


if __name__=='__main__':
   #CountId('/home/niezhaochang/ad-game/data/train.csv','/home/niezhaochang/ad-game/data/test1.csv')
   #StaticAdPositiveAndNegative('/home/niezhaochang/ad-game/data/train.csv','/home/niezhaochang/ad-game/data/test1.csv')
   #Interest_kw_topic()
   x_train,x_verify,y_train = runXGB(pd.read_csv('combine_train.csv',sep=','),pd.read_csv('combine_test.csv',sep=','))
   train_X,test_X,train_Y,test_Y=train_test_split(x_train, y_train, test_size=0.2, random_state=2018)
   model = xgb.XGBClassifier(learning_rate=0.1,
                             n_estimators=200,
                             max_depth=6,
                             min_child_weight = 1,
                             gamma=0,
                             subsample=0.8,
                             colsample_btree=0.8,
                             objective='binary:logistic',
                             scale_pos_weight=1,
                             random_state=27,
                             nthread = 20,
                             colsample_bylevel = 1,
                             reg_alpha = 0,
                             reg_lambda = 1)
   model.fit(train_X, train_Y, eval_metric='logloss', eval_set=[test_X,test_Y], verbose=True, early_stopping_rounds=100)
   y_pred = model.predict_proba(train_X)
   print(roc_auc_score(test_Y, y_pred))
   yres = model.predict(x_verify)
   print(yres)