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

train_train_datas,train_train_labels,headers = read_data_from_csv('../../data/gbdt_train_train.csv')
print (train_train_datas.shape, train_train_datas.shape,len(headers))
train_test_datas, train_test_labels,headers = read_data_from_csv('../../data/gbdt_train_test.csv')
test_datas, test_labels,headers = read_data_from_csv('../../data/gbdt_test.csv')




lgb_train_train = lgb.Dataset(train_train_datas, label=train_train_labels)
lgb_train_test = lgb.Dataset(train_test_datas, label=train_test_labels)

clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=127, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=2000, objective='binary',
    subsample=0.8, colsample_bytree=0.8, subsample_freq=1,
    learning_rate=0.08, min_child_weight=50
)
clf.fit(train_train_datas, train_train_labels, eval_set=[(train_train_datas, train_train_labels),(train_test_datas, train_test_labels)], eval_metric='logloss',early_stopping_rounds=100)

print(clf.feature_importances_)

res = clf.predict_proba(test_datas)[:,1]
f = open('../../data/res_x.csv','wb')
for r in res:
    f.write(str(r)+'\n')


