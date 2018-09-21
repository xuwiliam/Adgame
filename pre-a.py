# coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

import hashlib, csv, math, os, pickle, subprocess

def hashstr(str, nr_bins):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1

def gen_hashed_fm_feats(feats, nr_bins = int(1e+7)):
    feats = ['{0}:{1}:1'.format(field-1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats


one_hot_feature=['advertiserId','campaignId', 'creativeId','creativeSize','adCategoryId', 'productId', 'productType','uid_count',
                 'interest1_len','interest2_len']
len_feature = ['interest1','interest2']
combine_onehot_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os']

vector_feature=['appIdAction','appIdInstall','kw1','kw2','kw3','topic1','topic2','topic3']
combine_vector_feature = ['ct', 'marriageStatus', 'interest1','interest2','interest3','interest4','interest5']
clean_feature = ['topic2','topic3','kw1','kw2','kw3','appIdAction','appIdInstall']

def read_data(path):
    f = open(path, 'rb')
    features = f.readline().strip().split(',')
    dict = {}
    num = 0
    for line in f:
        datas = line.strip().split(',')
        for i, d in enumerate(datas):
            if not dict.has_key(features[i]):
                dict[features[i]] = []
            dict[features[i]].append(d)
        num += 1
    f.close()
    return dict,num

def transform_data(path, dict, num):
    ftrain = open(path, 'wb')
    flag = True
    max_len_dict = {}
    for i in range(num):
        feats = []
        for j, f in enumerate(one_hot_feature, 1):
            field = j
            if flag:
                print(field-1, f)
            feats.append((field, f + '_' + dict[f][i]))
        for j, f in enumerate(combine_onehot_feature, 1):
            field = j + len(one_hot_feature)
            xs = dict[f][i].split(' ')
            if flag:
                print(field-1, f)
            for x in xs:
                feats.append((field, 'aid_' + dict['aid'][i] + '_' + f + '_' + x))
        for j, f in enumerate(vector_feature, 1):
            field = j + len(one_hot_feature) + len(combine_onehot_feature)
            xs = dict[f][i].split(' ')
            if flag:
                print(field-1, f)
                max_len_dict[f] = len(xs)
            max_len_dict[f] = max(max_len_dict[f],len(xs))
            for x in xs:
                feats.append((field, f + '_' + x))
        for j, f in enumerate(combine_vector_feature, 1):
            field = j + len(one_hot_feature) + len(combine_onehot_feature) + len(vector_feature)
            xs = dict[f][i].split(' ')
            if flag:
                print(field-1, f)
                max_len_dict[f] = len(xs)
            max_len_dict[f] = max(max_len_dict[f], len(xs))
            for x in xs:
                feats.append((field, 'aid_' + dict['aid'][i] + '_' + f + '_' + x))
        flag = False
        feats = gen_hashed_fm_feats(feats)
        ftrain.write(dict['label'][i] + ' ' + ' '.join(feats) + '\n')
    for f in max_len_dict:
        print(f, max_len_dict[f])
    ftrain.close()

def clean(train_data, test_data):
    train_s = set()
    test_s = set()
    for i, data in enumerate(train_data):
        xs = data.split(' ')
        for x in xs:
            train_s.add(x)
    for i, data in enumerate(test_data):
        xs = data.split(' ')
        for x in xs:
            test_s.add(x)
    new_train_data = []
    new_test_data = []
    for i, data in enumerate(train_data):
        xs = data.split(' ')
        nxs = []
        for x in xs:
            if x in test_s:
                nxs.append(x)
        if len(nxs) == 0:
            nxs = ['-1']
        new_train_data.append(' '.join(nxs))
    for i, data in enumerate(test_data):
        xs = data.split(' ')
        nxs = []
        for x in xs:
            if x in train_s:
                nxs.append(x)
        if len(nxs) == 0:
            nxs = ['-1']
        new_test_data.append(' '.join(nxs))
    return new_train_data, new_test_data

def add_len(data):
    res = []
    for i, d in enumerate(data):
        if d=='-1':
            res.append('0')
        else:
            xs = d.split(' ')
            res.append(str(len(xs)))
    return res

def resort_feature(data):
    res = []
    for i, d in enumerate(data):
        xs = d.split(' ')
        tmp = []
        for x in xs:
            tmp.append(int(x))
        sorted(tmp)
        stmp = []
        for t in tmp:
            stmp.append(str(t))
        res.append(' '.join(stmp))
    return res

def add_user_count(train_data, test_data):
    total_dict = {}
    for d in train_data:
        if not total_dict.has_key(d):
            total_dict[d] = 0
        total_dict[d] += 1
    for d in test_data:
        if not total_dict.has_key(d):
            total_dict[d] = 0
        total_dict[d] += 1
    for d in total_dict:
        if total_dict[d] > 7:
            total_dict[d] = 7
    train_res = []
    for d in train_data:
        train_res.append(train_data[d])
    test_res = []
    for d in test_data:
        test_res.append(test_data[d])
    return train_res, test_res



print(len(one_hot_feature) + len(combine_onehot_feature))

print("reading train data")
train_dict, train_num = read_data('../../data/combine_train.csv')
print("reading test data")
test_dict, test_num = read_data('../../data/combine_test.csv')

print("adding len feature")
for f in len_feature:
    train_dict[f+'_len'] = add_len(train_dict[f])
    test_dict[f+'_len'] = add_len(test_dict[f])

print("cleaning data")
for f in clean_feature:
    train_dict[f], test_dict[f] = clean(train_dict[f], test_dict[f])

print("adding uid count")
train_dict['uid_count'], test_dict['uid_count'] = clean(train_dict['uid'], test_dict['uid'])

# print "resorting data"
# for f in combine_feature:
#     train_dict[f] = resort_feature(train_dict[f])
#     test_dict[f] = resort_feature(test_dict[f])


print("transforming train data")
transform_data('../../data/train.ffm',train_dict,train_num)
print("transforming test data")
transform_data('../../data/test.ffm',test_dict,test_num)
