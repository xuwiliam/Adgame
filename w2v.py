# coding=utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

from datetime import datetime
from gensim.models.word2vec import Word2Vec

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

print("reading train data")
train_dict, train_num = read_data('../../data/combine_train.csv')
print("reading test data")
test_dict, test_num = read_data('../../data/combine_test.csv')

def train_w2v(data):
    model = Word2Vec(data, sg=0, size=4, window=10, min_count=3, hs=0, workers=12)
    return model

interest_feature = ['interest1','interest2','interest3','interest4','interest5']

vector_feature = ['kw1','kw2','kw3','topic1','topic2','topic3']

print("training interest feature w2v model")
for f in interest_feature:
    data = []
    for i, item in enumerate(train_dict[f]):
        if train_dict['label'][i] == '1' and train_dict[f][i] != '-1':
            row = []
            xs = item.split(' ')
            for x in xs:
                row.append(train_dict['aid'][i]+'_'+x)
            data.append(row)
    print(f, len(data))
    model = train_w2v(data)
    model.save("../../data/w2v_model/"+f+"_pos_w2v.model")

print("training vector feature w2v model")
for f in vector_feature:
    sentences = []
    for i, item in enumerate(train_dict[f]):
        if train_dict['label'][i] == '1' and train_dict[f][i] != '-1':
            row = []
            xs = item.split(' ')
            for x in xs:
                row.append(x)
            sentences.append(row)
    print(f, len(sentences))
    model = train_w2v(sentences)
    model.save("../../data/w2v_model/"+f+"_pos_w2v.model")


