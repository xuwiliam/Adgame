# coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

import hashlib, csv, math, os, pickle, subprocess
import numpy as np
from gensim.models.word2vec import Word2Vec
import StaticsFeature
one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','aid','advertiserId','campaignId', 'creativeId','creativeSize','adCategoryId', 'productId', 'productType',
                 'interest1_len','interest2_len','interest3_len','interest4_len','interest5_len','kw1_len','kw2_len','kw3_len','topic1_len','topic2_len','topic3_len',
                 'marriageStatus', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5']
len_feature = ['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall']
vector_feature=['appIdAction','appIdInstall','kw1','kw2','kw3','topic1','topic2','topic3']
combine_feature = ['marriageStatus', 'interest1','interest2','interest3','interest4','interest5']
clean_feature = ['topic2','topic3','kw1','kw2','kw3','appIdAction','appIdInstall']
ratio_feature = ['advertiserId','campaignId','creativeId','creativeSize','adCategoryId','productId','productType',
             'LBS','age','appIdAction','appIdInstall','carrier','consumptionAbility','ct','education','gender','house','interest1',
             'interest2','interest3','interest4','interest5','kw1','kw2','kw3','marriageStatus','os','topic1','topic2','topic3']
def read_data(path):
    f = open(path, 'r')
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

def base_word2vec(x, model, size):
    vec = np.zeros(size)
    x = [item for item in x if model.wv.__contains__(item)]

    for item in x:
        vec += model.wv[item]
    if len(x) == 0:
        return vec
    else:
        return vec / len(x)

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
            res.append(0)
        else:
            xs = d.split(' ')
            res.append(len(xs))
    return res

def gen_count_dict(data, labels, begin, end):
    total_dict = {}
    pos_dict = {}
    for i, d in enumerate(data):
        if i >= begin and i < end:
            continue
        xs = d.split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0.0
            if not pos_dict.has_key(x):
                pos_dict[x] = 0.0
            total_dict[x] += 1
            if labels[i] == '1':
                pos_dict[x] += 1
    return total_dict, pos_dict

def gen_combine_count_dict(data1, data2, labels, begin, end):
    total_dict = {}
    pos_dict = {}
    for i, d in enumerate(data1):
        if i >= begin and i < end:
            continue
        xs = d.split(' ')
        xs2 = data2[i].split(' ')
        for x1 in xs:
            for x2 in xs2:
                k = x1+'|'+x2
                if not total_dict.has_key(k):
                    total_dict[k] = 0.0
                if not pos_dict.has_key(k):
                    pos_dict[k] = 0.0
                total_dict[k] += 1
                if labels[i] == '1':
                    pos_dict[k] += 1
    return total_dict, pos_dict


def count_feature(train_data, test_data, labels, k, typ=True):
    nums = 8798814
    interval = nums // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(nums)

    s = set()
    for d in train_data:
        xs = d.split(' ')
        for x in xs:
            s.add(x)
    b = nums // len(s)
    a = b*1.0 / 20

    train_res = []
    for i in range(k):
        tmp = []
        total_dict, pos_dict = gen_count_dict(train_data, labels, split_points[i],split_points[i+1])
        for j in range(split_points[i],split_points[i+1]):
            xs = train_data[j].split(' ')
            t = 0.0
            for x in xs:
                if not total_dict.has_key(x):
                    t += 0.05
                    continue
                t += (a + pos_dict[x]) / (b + total_dict[x])
            if typ:
                tmp.append(t / len(xs))
            else:
                tmp.append(t)

        train_res.extend(tmp)

    test_res = []
    total_dict, pos_dict = gen_count_dict(train_data, labels, 1, 0)
    for d in test_data:
        xs = d.split(' ')
        t = 0.0
        for x in xs:
            if not total_dict.has_key(x):
                t += 0.05
                continue
            t += (a + pos_dict[x]) / (b + total_dict[x])
        if typ:
            test_res.append(t / len(xs))
        else:
            test_res.append(t)

    return train_res, test_res


def count_combine_feature(train_data1, train_data2, test_data1, test_data2, labels, k, typ = True):
    nums = 8798814
    interval = nums // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(nums)

    s = set()
    for i, d in enumerate(train_data1):
        xs = d.split(' ')
        xs2 = train_data2[i].split(' ')
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                s.add(ke)
    b = nums // len(s)
    a = b*1.0 / 20

    train_res = []
    for i in range(k):
        tmp = []
        total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2,labels, split_points[i],split_points[i+1])
        for j in range(split_points[i],split_points[i+1]):
            xs = train_data1[j].split(' ')
            xs2 = train_data2[j].split(' ')
            t = 0.0
            c = 0
            for x1 in xs:
                for x2 in xs2:
                    c += 1
                    ke = x1 + '|' + x2
                    if not total_dict.has_key(ke):
                        t += 0.05
                        continue
                    t += (a + pos_dict[ke]) / (b + total_dict[ke])
            if typ:
                tmp.append(t / c)
            else:
                tmp.append(t)
        train_res.extend(tmp)

    test_res = []
    total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2, labels, 1, 0)
    for i,d in enumerate(test_data1):
        xs = d.split(' ')
        xs2 = test_data2[i].split(' ')
        t = 0.0
        c = 0
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                c += 1
                if not total_dict.has_key(ke):
                    t += 0.05
                    continue
                t += (a + pos_dict[ke]) / (b + total_dict[ke])
        if typ:
            test_res.append(t / c)
        else:
            test_res.append(t)

    return train_res, test_res

def add_w2v_feature(data, model, ad_data = None):
    arr = []
    for i in range(4):
        arr.append([])
    for j,d in enumerate(data):
        xs = d.split(' ')
        if ad_data:
            for i,x in enumerate(xs):
                xs[i] = ad_data[j] + '_' + x
        ones = base_word2vec(xs, model, 4)
        for i, one in enumerate(ones):
            arr[i].append(one)
    return arr

user_combine_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1','interest2','interest3','interest4','interest5',
                        'marriageStatus']
user_one_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1','interest2','interest3','interest4','interest5',
                        'marriageStatus','topic1','topic2','topic3','kw1','kw2','kw3','appIdAction','appIdInstall']
ad_feature = ['advertiserId','campaignId', 'creativeId','creativeSize','adCategoryId', 'productId', 'productType']

w2v_interest_feature = ['interest1','interest2','interest3','interest4','interest5']
w2v_vector_feature = ['kw1','kw2','kw3','topic1','topic2','topic3']


print("reading train data")
train_dict, train_num = read_data('data/small_combine_train.txt')
print("reading test data")
test_dict, test_num = read_data('data/small_combine_test.txt')

print(train_num, test_num)

headers = ['label']
print("adding len feature")
for f in len_feature:
    print (f+'_len adding')
    train_dict[f+'_len'] = add_len(train_dict[f])
    test_dict[f+'_len'] = add_len(test_dict[f])
    headers.append(f+'_len')


# print ("adding interest w2v feature")
# for f in w2v_interest_feature:
#     print (f+"_pos_w2v adding")
#     model = Word2Vec.load('../../data/w2v_model/'+f+'_pos_w2v.model')
#     train_dict[f+'_pos_w2v_1'],train_dict[f+'_pos_w2v_2'],train_dict[f+'_pos_w2v_3'],train_dict[f+'_pos_w2v_4'] = add_w2v_feature(train_dict[f],model,train_dict['aid'])
#     test_dict[f + '_pos_w2v_1'], test_dict[f + '_pos_w2v_2'], test_dict[f + '_pos_w2v_3'], test_dict[
#         f + '_pos_w2v_4'] = add_w2v_feature(test_dict[f], model, test_dict['aid'])
#     headers.append(f + '_pos_w2v_1')
#     headers.append(f + '_pos_w2v_2')
#     headers.append(f + '_pos_w2v_3')
#     headers.append(f + '_pos_w2v_4')
#
# print ("adding vector w2v feature")
# for f in w2v_vector_feature:
#     print (f+"_w2v adding")
#     model = Word2Vec.load('../../data/w2v_model/'+f+'_w2v.model')
#     train_dict[f+'_w2v_1'],train_dict[f+'_w2v_2'],train_dict[f+'_w2v_3'],train_dict[f+'_w2v_4'] = add_w2v_feature(train_dict[f],model)
#     test_dict[f + '_w2v_1'], test_dict[f + '_w2v_2'], test_dict[f + '_w2v_3'], test_dict[
#         f + '_w2v_4'] = add_w2v_feature(test_dict[f], model)
#     headers.append(f + '_w2v_1')
#     headers.append(f + '_w2v_2')
#     headers.append(f + '_w2v_3')
#     headers.append(f + '_w2v_4')

print("cleaning data")
for f in clean_feature:
    train_dict[f], test_dict[f] = clean(train_dict[f], test_dict[f])

print( "adding one user feature")
for f in user_one_feature:
    print(f+'_count adding')
    train_res, test_res = count_feature(train_dict[f],test_dict[f],train_dict['label'], 5)#单独取出每一列统计
    train_dict[f+'_count'] = train_res
    test_dict[f+'_count'] = test_res
    headers.append(f+'_count')

print ("adding combine feature")
for f in user_combine_feature:
    print(f+'_aid_count adding')
    train_res, test_res = count_combine_feature(train_dict[f],train_dict['aid'],test_dict[f],test_dict['aid'],train_dict['label'],5)
    train_dict[f+'_aid_count'] = train_res
    test_dict[f+'_aid_count'] = test_res
    headers.append(f+'_aid_count')

print( "adding ad feature")
for f in ad_feature:

    print( f+'_count adding')
    train_res, test_res = count_feature(train_dict[f],test_dict[f],train_dict['label'], 5)
    train_dict[f+'_count'] = train_res
    test_dict[f+'_count'] = test_res
    headers.append(f+'_count')

cnt = 0
df = [0] * 31
df2 = [0] * 31
dict_ = {}
train_dict = pd.read_csv('combine_train.csv', sep=',')
test_dict = pd.read_csv('combine_test.csv', sep=',')
for f in ratio_feature:
    df[cnt] = pd.read_csv(f + '_count', sep=',', header=None)
    df2[cnt] = pd.DataFrame(df[cnt].values, columns=['value', 'ratio'])
    print(df2[cnt])
    print(f)
    for index, row in df2[cnt].iterrows():
        print(row['value'], row['ratio'])
        dict_[str(int(row['value']))] = float(row['ratio'])
    print(dict_)
    train_res, test_res = StaticsFeature.InsertFeature(train_dict[f], test_dict[f], dict_, f)
    train_dict[f + '_ratio'] = train_res
    test_dict[f + '_ratio'] = test_res
    headers.append(f + '_ratio')
    cnt += 1

print ("writing train data")
fw = open('data/gbdt_train.csv','wb')
fw.write(','.join(headers)+'\n')
for i in range(train_num):
    row = [train_dict['label'][i]]
    for f in headers[1:]:
        t = '%.6f' % float(train_dict[f][i])
        row.append(t)
    fw.write(','.join(row)+'\n')
fw.close()

print("writing test data")
fw = open('data/gbdt_test.csv','wb')
fw.write(','.join(headers)+'\n')
for i in range(test_num):
    row = [test_dict['label'][i]]
    for f in headers[1:]:
        t = '%.6f' % float(test_dict[f][i])
        row.append(t)
    fw.write(','.join(row)+'\n')
fw.close()