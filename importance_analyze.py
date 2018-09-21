# coding=utf-8
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os
import numpy as np
from time import time

data_root_path = '/home/disk2/niezhaochang/ad-game/data/statistics/'
analyze_root_path = '/home/disk2/niezhaochang/ad-game/data/analyze/'
user_combine_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1','interest2','interest3','interest4','interest5', 'marriageStatus']
user_combine_feature_2 = ['topic1', 'topic2', 'topic3', 'kw1', 'kw2', 'kw3', 'appIdAction','appIdInstall']

user_one_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1','interest2','interest3','interest4','interest5',
                        'marriageStatus','topic1','topic2','topic3','kw1','kw2','kw3','appIdAction','appIdInstall']

ad_feature = ['advertiserId','campaignId', 'creativeId','creativeSize','adCategoryId', 'productId', 'productType']

len_feature = ['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall']

col_names = ['label']
for f in len_feature:
    col_names.append(f+'_len')

for f in user_one_feature:
    col_names.append(f+'_count')

for f in user_combine_feature:
    col_names.append(f+'_aid_count')

for f in user_combine_feature_2:
    col_names.append(f+'_aid_count')

for f in ad_feature:
    col_names.append(f+'_count')

for f in user_combine_feature:
    col_names.append(f+'_aid_cross')
    col_names.append('aid_' + f + '_cross')

# for f in user_combine_feature:
#     col_names.append(f + '_aid_pos_cross')
#     col_names.append('aid_' + f + '_pos_cross')

# for i, f in enumerate(user_combine_feature):
#     for tf in user_combine_feature[i+1:]:
#         col_names.append(f+'_'+tf+'_count')

col_names.append('uid_count')
col_names.append('uid_is_rapeat')


def read_data_from_bin(path, col_names, nums = 7000000):
    data_arr = []
    for col_name in col_names:
        data_arr.append(np.fromfile(path + col_name +'.bin', dtype=np.float)[0:nums])
    return np.concatenate([data_arr],axis=1).T

def read_all_data_from_bin(path, col_names, train_indexs, test_indexs):
    data_arr = []
    for col_name in col_names:
        data_arr.append(np.fromfile(path + col_name +'.bin', dtype=np.float))
    all = np.concatenate([data_arr],axis=1).T
    return all[train_indexs], all[test_indexs]

def read_indexs(path):
    f = open(path,'rb')
    res = []
    for line in f:
        res.append(int(line.strip())-1)
    return res

train_train_indexs = read_indexs('../../data/train_train_indexs.csv')
train_test_indexs = list(set(range(0,8798814)).difference(set(train_train_indexs)))

train_train_labels, train_test_labels = read_all_data_from_bin(data_root_path+'train/',col_names[0:1],train_train_indexs,train_test_indexs)
train_train_labels = train_train_labels.reshape([-1])
train_test_labels = train_test_labels.reshape([-1])
train_train_datas, train_test_datas = read_all_data_from_bin(data_root_path+'train/',col_names[1:],train_train_indexs,train_test_indexs)
print (train_train_datas.shape, train_train_labels.shape)




clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=63, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=2000, objective='binary',
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1, feature_fraction=0.8,
    learning_rate=0.05, min_child_weight=50
)
clf.fit(train_train_datas, train_train_labels, eval_set=[(train_train_datas, train_train_labels),(train_test_datas, train_test_labels)], eval_metric='logloss',early_stopping_rounds=100)

dict = {}
for i,col_name in enumerate(col_names[1:]):
    dict[col_name] = clf.feature_importances_[i] * 100

f = open(analyze_root_path + 'importance_rank.csv','wb')
f.write('name,value\n')
sort_dict = sorted(dict.items(),key = lambda x:x[1],reverse = True)
for name, val in sort_dict:
    f.write(name + ',' + str(val) + '\n')
f.close()

test_datas = read_data_from_bin(data_root_path + 'test/', col_names[1:])
res = clf.predict_proba(test_datas)[:,1]
f = open('../../data/res.csv','wb')
for r in res:
    f.write(str(r)+'\n')


