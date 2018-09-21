import xgboost as xgb
import sys
from sklearn.metrics import roc_auc_score
from random import random
import numpy as np


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
print(train_train_datas.shape, train_train_labels.shape)

dtrain = xgb.DMatrix(train_train_datas, train_train_labels)
dtest = xgb.DMatrix(train_test_datas, train_test_labels)


param = {}
param['objective'] = 'binary:logistic'
param['booster'] = 'gbtree'
param['eta'] = 0.01
param['max_depth'] = 8
param['silent'] = 1
param['nthread'] = 16
param['subsample'] = 0.8
param['colsample_bytree'] = 0.8
param['eval_metric'] = 'logloss'
num_round = 1500

param['tree_method'] = 'exact'
param['scale_pos_weight'] = 1
watchlist = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(param, dtrain, num_round, watchlist)

test_datas = read_data_from_bin(data_root_path + 'test/', col_names[1:])
res = xgb.predict(test_datas)

f = open('../../data/res.csv','wb')
for r in res:
    f.write(str(r)+'\n')
