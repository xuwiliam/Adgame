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

def read_combine_data(path,max_num = 100000000):
    f = open(path, 'rb')
    features = f.readline().strip().split(',')
    dict = {}
    num = 0
    for line in f:
        num += 1
        if num > max_num:
            break
        datas = line.strip().split(',')
        for i, d in enumerate(datas):
            if not dict.has_key(features[i]):
                dict[features[i]] = []
            dict[features[i]].append(d)
    f.close()
    return dict,num

def read_indexs(path):
    f = open(path,'rb')
    res = []
    for line in f:
        res.append(int(line.strip())-1)
    return res

def read_data_from_bin(path, col_names):
    data_arr = {}
    for col_name in col_names:
        data_arr[col_name] = np.fromfile(path + col_name +'.bin', dtype=np.float)
    return data_arr

def get_top_count_feature(path, num=50):
    col_names = []
    f = open(path, 'rb')
    f.readline()
    i = 0
    for line in f:
        i += 1
        if i > num:
            break
        col_names.append(line.strip().split(',')[0])
    return col_names

def read_data_from_csv(path):
    f = open(path, 'rb')
    headers = f.readline().strip().split(',')
    datas = []
    labels = []
    i = 0
    for line in f:
        i += 1
        ds = line.strip().split(',')
        labels.append(int(ds[0]))
        row = []
        for d in ds[1:]:
            row.append(float(d))
        datas.append(row)
    f.close()
    return np.array(datas), np.array(labels),headers

print ("reading train data")
train, train_num = read_combine_data('../../data/combine_train.csv')
print ("reading test data")
test, test_num = read_combine_data('../../data/combine_test.csv')

vector_feature=['label','appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
train = pd.DataFrame(train)[vector_feature]
test = pd.DataFrame(test)[vector_feature]
data=pd.concat([train,test])

train_train_indexs = read_indexs('../../data/train_train_indexs.csv')
train_test_indexs = list(set(range(0,train.shape[0])).difference(set(train_train_indexs)))
train_train = train.iloc[train_train_indexs]
train_test = train.iloc[train_test_indexs]
train_y=train.pop('label')
train_train_y = train_y.iloc[train_train_indexs]
train_test_y = train_y.iloc[train_test_indexs]
test=test.drop('label',axis=1)

print("loading count feature")
data_root_path = '/home/disk2/niezhaochang/ad-game/data/statistics/'
analyze_root_path = '/home/disk2/niezhaochang/ad-game/data/analyze/'
col_names = get_top_count_feature(analyze_root_path + 'importance_rank.csv',80)
col_names.append('uid_count')
train_x = read_data_from_bin(data_root_path + 'train/', col_names)
test_x = read_data_from_bin(data_root_path + 'test/', col_names)
train_x = pd.DataFrame(train_x)
train_train_x = train_x.iloc[train_train_indexs]
train_test_x = train_x.iloc[train_test_indexs]
test_x = pd.DataFrame(test_x)

cv=CountVectorizer(max_features=300)
for feature in vector_feature[1:]:
    print (feature, 'preparing')
    cv.fit(data[feature])
    train_train_a = cv.transform(train_train[feature])
    train_test_a = cv.transform(train_test[feature])
    train_train_x = sparse.hstack((train_train_x, train_train_a))
    train_test_x = sparse.hstack((train_test_x, train_test_a))
    test_a = cv.transform(test[feature])
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')


clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=127, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=2000, objective='binary',
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1,feature_fraction=0.8,
    learning_rate=0.08, min_child_weight=50
)
clf.fit(train_train_x, train_train_y, eval_set=[(train_train_x, train_train_y),(train_test_x, train_test_y)], eval_metric='logloss',early_stopping_rounds=100)

print( clf.feature_importances_)

res = clf.predict_proba(test_x)[:,1]
f = open('../../data/res.csv','wb')
for r in res:
    f.write(str(r)+'\n')


