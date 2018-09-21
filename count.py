# coding=utf-8

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import os

ad_feature=pd.read_csv('../../data/adFeature.csv')
if os.path.exists('../../data/userFeature.csv'):
    user_feature=pd.read_csv('../../data/userFeature.csv')
else:
    userFeature_data = []
    with open('../../data/userFeature.data', 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('|')
            userFeature_dict = {}
            for each in line:
                each_list = each.split(' ')
                userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
            userFeature_data.append(userFeature_dict)
            if i % 100000 == 0:
                print(i)
        user_feature = pd.DataFrame(userFeature_data)
        user_feature.to_csv('../../data/userFeature.csv', index=False)
train=pd.read_csv('../../data/train.csv')
predict=pd.read_csv('../../data/test1.csv')
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('-1')

train = data[data.label != -1]
pos_train = data[data.label == 1]
neg_train = data[data.label == 0]
test = data[data.label == -1]

ad_features = ['aid']

#user_features = ['age', 'gender', 'marriageStatus', 'education', 'consumptionAbility', 'LBS', 'ct', 'os', 'carrier', 'house']
user_features = ['interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']

# def count_data(adata, data, idx, counter):
#     total = 0
#     for i, d in enumerate(data):
#         d = str(d)
#         xs = d.split(' ')
#         for x in xs:
#             k = str(adata.iloc[i]) + '|' + x
#             if k not in counter:
#                 counter[k] = [0.0, 0.0]
#             counter[k][idx] += 1
#         total += 1
#     return counter, total
#
# for fa in ad_features:
#     for f in user_features:
#         counter = {}
#         counter, pos_total = count_data(pos_train[fa], pos_train[f], 0, counter)
#         counter, neg_total = count_data(neg_train[fa], neg_train[f], 1, counter)
#         print fa + ' ----- ' + f
#         for x in counter:
#             print x, counter[x][0]/pos_total, counter[x][1]/neg_total
#             print '-'*10

def count_max(data):
    t = 0
    s = set()
    for d in data:
        xs = d.split(' ')
        for x in xs:
            s.add(x)
    return len(s)

for f in user_features:
    print(f, count_max(data[f]))



