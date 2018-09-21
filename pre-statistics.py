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

root_path = '/home/disk2/niezhaochang/ad-game/data/statistics/'

one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','aid','advertiserId','campaignId', 'creativeId','creativeSize','adCategoryId', 'productId', 'productType',
                 'interest1_len','interest2_len','interest3_len','interest4_len','interest5_len','kw1_len','kw2_len','kw3_len','topic1_len','topic2_len','topic3_len',
                 'marriageStatus', 'interest1', 'interest2', 'interest3', 'interest4', 'interest5']

vector_feature=['appIdAction','appIdInstall','kw1','kw2','kw3','topic1','topic2','topic3']
combine_feature = ['marriageStatus', 'interest1','interest2','interest3','interest4','interest5']
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

def write_data_by_col(path, data):
    data = np.array(data)
    data.tofile(path)


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
            res.append(0.0)
        else:
            xs = d.split(' ')
            res.append(float(len(xs)))
    return res

def gen_count_dict(data, labels, indexs, begin, end):
    total_dict = {}
    pos_dict = {}
    flags = [1] * len(indexs)
    for i in indexs[begin:end]:
        flags[i] = 0
    for i in indexs:
        if not flags[i]:
            continue
        xs = data[i].split(' ')
        for x in xs:
            if not total_dict.has_key(x):
                total_dict[x] = 0.0
            if not pos_dict.has_key(x):
                pos_dict[x] = 0.0
            total_dict[x] += 1
            if labels[i] == '1':
                pos_dict[x] += 1
    return total_dict, pos_dict

def gen_combine_count_dict(data1, data2, labels, indexs, begin, end):
    total_dict = {}
    pos_dict = {}
    flags = [1]*len(indexs)
    for i in indexs[begin:end]:
        flags[i] = 0
    for i in indexs:
        if not flags[i]:
            continue
        xs = data1[i].split(' ')
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

    #shuffle data
    indexs = range(nums)
    np.random.shuffle(indexs)

    s = set()
    for d in train_data:
        xs = d.split(' ')
        for x in xs:
            s.add(x)
    b = nums // len(s)
    a = b*1.0 / 20

    train_res = [0.0]*nums
    for i in range(k):
        total_dict, pos_dict = gen_count_dict(train_data, labels, indexs, split_points[i],split_points[i+1])
        for j in range(split_points[i],split_points[i+1]):
            xs = train_data[indexs[j]].split(' ')
            t = 0.0
            for x in xs:
                if not total_dict.has_key(x):
                    t += 0.05
                    continue
                t += (a + pos_dict[x]) / (b + total_dict[x])
            if typ:
                train_res[indexs[j]] = t / len(xs)
            else:
                train_res[indexs[j]] = t


    test_res = [0.0]*len(test_data)
    total_dict, pos_dict = gen_count_dict(train_data, labels, indexs, 1, 0)
    for i, d in enumerate(test_data):
        xs = d.split(' ')
        t = 0.0
        for x in xs:
            if not total_dict.has_key(x):
                t += 0.05
                continue
            t += (a + pos_dict[x]) / (b + total_dict[x])
        if typ:
            test_res[i] = t / len(xs)
        else:
            test_res[i] = t

    return train_res, test_res


def count_combine_feature(train_data1, train_data2, test_data1, test_data2, labels, k, typ = True):
    nums = 8798814
    interval = nums // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(nums)

    #shuffle data
    indexs = range(nums)
    np.random.shuffle(indexs)


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

    train_res = [0.0]*nums
    for i in range(k):
        total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2, labels, indexs, split_points[i], split_points[i+1])
        for j in range(split_points[i],split_points[i+1]):
            xs = train_data1[indexs[j]].split(' ')
            xs2 = train_data2[indexs[j]].split(' ')
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
                train_res[indexs[j]] = t / c
            else:
                train_res[indexs[j]] = t

    test_res = [0.0] * len(test_data1)
    total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2, labels, indexs, 1, 0)
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
            test_res[i] = t / c
        else:
            test_res[i] = t

    return train_res, test_res

def count_cross_feature(train_data1, train_data2, test_data1, test_data2, labels, k):
    nums = 8798814
    indexs = range(nums)
    interval = nums // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(nums)

    #shullfe data
    # rng_state = np.random.get_state()
    np.random.shuffle(indexs)
    # np.random.set_state(rng_state)
    # np.random.shuffle(train_data1)
    # np.random.set_state(rng_state)
    # np.random.shuffle(train_data2)
    # np.random.set_state(rng_state)
    # np.random.shuffle(test_data1)
    # np.random.set_state(rng_state)
    # np.random.shuffle(test_data2)
    # np.random.set_state(rng_state)
    # np.random.shuffle(labels)

    #train data count
    train_res_1 = [0.0]*nums
    train_res_2 = [0.0]*nums
    for i in range(k):
        total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2, labels, indexs, split_points[i],
                                                      split_points[i + 1])
        total_dict_1, pos_dict_1 = gen_count_dict(train_data1, labels, indexs, split_points[i],split_points[i+1])
        total_dict_2, pos_dict_2 = gen_count_dict(train_data2, labels, indexs, split_points[i],split_points[i+1])
        for j in range(split_points[i], split_points[i + 1]):
            xs = train_data1[indexs[j]].split(' ')
            xs2 = train_data2[indexs[j]].split(' ')
            t_1 = 0.0 # (f1,f2)/f1
            t_2 = 0.0 # (f1,f2)/f2
            c = 0
            for x1 in xs:
                for x2 in xs2:
                    ke = x1 + '|' + x2
                    if not total_dict.has_key(ke):
                        continue
                    c += 1
                    t_1 += total_dict[ke] / total_dict_1[x1]
                    t_2 += total_dict[ke] / total_dict_2[x2]
                if c != 0:
                    train_res_1[indexs[j]] = t_1 / c
                    train_res_2[indexs[j]] = t_2 / c

    #test data count
    test_res_1 = [0.0]*len(test_data1)
    test_res_2 = [0.0]*len(test_data2)
    total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2, labels, indexs, 1, 0)
    total_dict_1, pos_dict_1 = gen_count_dict(train_data1, labels, indexs, 1, 0)
    total_dict_2, pos_dict_2 = gen_count_dict(train_data2, labels, indexs, 1, 0)
    for i, d in enumerate(test_data1):
        xs = d.split(' ')
        xs2 = test_data2[i].split(' ')
        t_1 = 0.0
        t_2 = 0.0
        c = 0
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                if not total_dict.has_key(ke):
                    continue
                c += 1
                t_1 += total_dict[ke] / total_dict_1[x1]
                t_2 += total_dict[ke] / total_dict_2[x2]
        if c != 0:
            test_res_1[i] = t_1 / c
            test_res_2[i] = t_2 / c

    return train_res_1, train_res_2, test_res_1, test_res_2

def count_pos_cross_feature(train_data1, train_data2, test_data1, test_data2, labels, k):
    nums = 8798814
    indexs = range(nums)
    interval = nums // k
    split_points = []
    for i in range(k):
        split_points.append(i * interval)
    split_points.append(nums)

    #shullfe data
    # rng_state = np.random.get_state()
    np.random.shuffle(indexs)
    # np.random.set_state(rng_state)
    # np.random.shuffle(train_data1)
    # np.random.set_state(rng_state)
    # np.random.shuffle(train_data2)
    # np.random.set_state(rng_state)
    # np.random.shuffle(test_data1)
    # np.random.set_state(rng_state)
    # np.random.shuffle(test_data2)
    # np.random.set_state(rng_state)
    # np.random.shuffle(labels)

    #train data count
    train_res_1 = [0.0]*nums
    train_res_2 = [0.0]*nums
    for i in range(k):
        total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2, labels, indexs, split_points[i],
                                                      split_points[i + 1])
        total_dict_1, pos_dict_1 = gen_count_dict(train_data1, labels, indexs, split_points[i],split_points[i+1])
        total_dict_2, pos_dict_2 = gen_count_dict(train_data2, labels, indexs, split_points[i],split_points[i+1])
        for j in range(split_points[i], split_points[i + 1]):
            xs = train_data1[indexs[j]].split(' ')
            xs2 = train_data2[indexs[j]].split(' ')
            t_1 = 0.0 # (f1,f2)/f1
            t_2 = 0.0 # (f1,f2)/f2
            c = 0
            for x1 in xs:
                for x2 in xs2:
                    ke = x1 + '|' + x2
                    if not pos_dict.has_key(ke):
                        continue
                    c += 1
                    t_1 += pos_dict[ke] / (pos_dict_1[x1] + 0.00001)
                    t_2 += pos_dict[ke] / (pos_dict_2[x2] + 0.00001)
                if c != 0:
                    train_res_1[indexs[j]] = t_1 / c
                    train_res_2[indexs[j]] = t_2 / c

    #test data count
    test_res_1 = [0.0]*len(test_data1)
    test_res_2 = [0.0]*len(test_data2)
    total_dict, pos_dict = gen_combine_count_dict(train_data1, train_data2, labels, indexs, 1, 0)
    total_dict_1, pos_dict_1 = gen_count_dict(train_data1, labels, indexs, 1, 0)
    total_dict_2, pos_dict_2 = gen_count_dict(train_data2, labels, indexs, 1, 0)
    for i, d in enumerate(test_data1):
        xs = d.split(' ')
        xs2 = test_data2[i].split(' ')
        t_1 = 0.0
        t_2 = 0.0
        c = 0
        for x1 in xs:
            for x2 in xs2:
                ke = x1 + '|' + x2
                if not pos_dict.has_key(ke):
                    continue
                c += 1
                t_1 += pos_dict[ke] / (pos_dict_1[x1] + 0.00001)
                t_2 += pos_dict[ke] / (pos_dict_2[x2] + 0.00001)
        if c != 0:
            test_res_1[i] = t_1 / c
            test_res_2[i] = t_2 / c

    return train_res_1, train_res_2, test_res_1, test_res_2


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

def count_user_times(train_data, test_data):
    total_dict = {}
    train_total_dict = {}
    train_s = set()
    test_s = set()
    for i, d in enumerate(train_data):
        if not total_dict.has_key(d):
            total_dict[d] = 0.0
            train_total_dict[d] = 0.0
            train_s.add(d)
        total_dict[d] += 1
        train_total_dict[d] += 1
    for i, d in enumerate(test_data):
        if not total_dict.has_key(d):
            total_dict[d] = 0.0
            test_s.add(d)
        total_dict[d] += 1

    train_res = []
    train_rapid = []
    for d in train_data:
        train_res.append(train_total_dict[d])
        if d in test_s:
            train_rapid.append(1.0)
        else:
            train_rapid.append(0.0)

    test_res = []
    test_rapid = []
    for d in test_data:
        test_res.append(total_dict[d])
        if d in train_s:
            test_rapid.append(1.0)
        else:
            test_rapid.append(0.0)
    return train_res, test_res,train_rapid,test_rapid

# user_combine_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1','interest2','interest3','interest4','interest5',
#                         'marriageStatus','topic1','topic2','topic3','kw1','kw2','kw3','appIdAction','appIdInstall']

user_combine_feature = ['topic1','topic2','topic3','kw1','kw2','kw3','appIdAction','appIdInstall']


user_one_feature = ['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','interest1','interest2','interest3','interest4','interest5',
                        'marriageStatus','topic1','topic2','topic3','kw1','kw2','kw3','appIdAction','appIdInstall']

ad_feature = ['advertiserId','campaignId', 'creativeId','creativeSize','adCategoryId', 'productId', 'productType']

len_feature = ['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdAction','appIdInstall']

w2v_interest_feature = ['interest1','interest2','interest3','interest4','interest5']

w2v_vector_feature = ['kw1','kw2','kw3','topic1','topic2','topic3']


print("reading train data")
train_dict, train_num = read_data('../../data/combine_train.csv')
print("reading test data")
test_dict, test_num = read_data('../../data/combine_test.csv')

print(train_num, test_num)
#
# print "adding label"
# labels = []
# for label in train_dict['label']:
#     labels.append(float(label))
# write_data_by_col(root_path+'train/label.bin',labels)




# print "adding len feature"
# for f in len_feature:
#     print f+'_len adding'
#     train_res = add_len(train_dict[f])
#     write_data_by_col(root_path+'train/'+f+'_len.bin', train_res)
#     test_res = add_len(test_dict[f])
#     write_data_by_col(root_path + 'test/' + f + '_len.bin', test_res)

#

# print "cleaning data"
# for f in clean_feature:
#     train_dict[f], test_dict[f] = clean(train_dict[f], test_dict[f])
#
# print "adding one user feature"
# for f in user_one_feature:
#     print f+'_count adding'
#     train_res, test_res = count_feature(train_dict[f],test_dict[f],train_dict['label'], 5)
#     write_data_by_col(root_path + 'train/' + f + '_count.bin', train_res)
#     write_data_by_col(root_path + 'test/' + f + '_count.bin', test_res)
#
# print "adding combine feature"
# for f in user_combine_feature:
#     print f+'_aid_count adding'
#     train_res, test_res = count_combine_feature(train_dict[f],train_dict['aid'],test_dict[f],test_dict['aid'],train_dict['label'],5)
#     write_data_by_col(root_path + 'train/' + f + '_aid_count.bin', train_res)
#     write_data_by_col(root_path + 'test/' + f + '_aid_count.bin', test_res)
#
# print "adding ad feature"
# for f in ad_feature:
#     print f+'_count adding'
#     train_res, test_res = count_feature(train_dict[f],test_dict[f],train_dict['label'], 5)
#     write_data_by_col(root_path + 'train/' + f + '_count.bin', train_res)
#     write_data_by_col(root_path + 'test/' + f + '_count.bin', test_res)
#
# print "adding cross feature"
# for f in user_combine_feature:
#     print f + '_aid_corss adding'
#     train_res_1, train_res_2, test_res_1, test_res_2 = count_cross_feature(train_dict[f],train_dict['aid'],test_dict[f],test_dict['aid'],train_dict['label'],5)
#     write_data_by_col(root_path + 'train/' + f + '_aid_cross.bin', train_res_1)
#     write_data_by_col(root_path + 'test/' + f + '_aid_cross.bin', test_res_1)
#     write_data_by_col(root_path + 'train/aid_' + f + '_cross.bin', train_res_2)
#     write_data_by_col(root_path + 'test/aid_' + f + '_cross.bin', test_res_2)

# print "adding pos cross feature"
# for f in user_combine_feature:
#     print f + '_aid_pos_corss adding'
#     train_res_1, train_res_2, test_res_1, test_res_2 = count_pos_cross_feature(train_dict[f],train_dict['aid'],test_dict[f],test_dict['aid'],train_dict['label'],5)
#     write_data_by_col(root_path + 'train/' + f + '_aid_pos_cross.bin', train_res_1)
#     write_data_by_col(root_path + 'test/' + f + '_aid_pos_cross.bin', test_res_1)
#     write_data_by_col(root_path + 'train/aid_' + f + '_pos_cross.bin', train_res_2)
#     write_data_by_col(root_path + 'test/aid_' + f + '_pos_cross.bin', test_res_2)



# print "adding user inner combine feature"
# for i, f in enumerate(user_combine_feature):
#     for tf in user_combine_feature[i+1:]:
#         print f+'_'+tf+'_count adding'
#         train_res, test_res = count_combine_feature(train_dict[f],train_dict[tf],test_dict[f],test_dict[tf],train_dict['label'],5)
#         write_data_by_col(root_path + 'train/' + f + '_' + tf + '_count.bin', train_res)
#         write_data_by_col(root_path + 'test/' + f + '_' + tf + '_count.bin', test_res)


print("uid_count adding")
train_res, test_res,train_rapid, test_rapid = count_user_times(train_dict['uid'], test_dict['uid'])
write_data_by_col(root_path+'train/uid_count.bin', train_res)
write_data_by_col(root_path+'test/uid_count.bin', test_res)
write_data_by_col(root_path+'train/uid_is_rapeat.bin', train_rapid)
write_data_by_col(root_path+'test/uid_is_rapeat.bin', test_rapid)
