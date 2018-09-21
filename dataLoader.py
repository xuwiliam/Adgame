#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 niezhaochang <niezhaochang@amax3>
#
# Distributed under terms of the MIT license.

"""
load data
"""

import pandas as pd
import os

AD_CSV = 'data/adFeature.csv'
USER_CSV = 'data/userFeature.csv'
USER_DATA = 'data/userFeature.data'
TRAIN_CSV = 'data/train.csv'
TEST_CSV = 'data/test1.csv'


def load_data():
    '''
    return: merge之后的训练集和测试集的pandas对象。
    '''
    ad_feature = pd.read_csv(AD_CSV)
    if os.path.exists(USER_CSV):
        user_feature = pd.read_csv(USER_CSV)
    else:
        userFeature_data = []
        with open(USER_DATA, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip().split('|')
                userFeature_dict = {}
                print('userFeature_dict: ')
                for each in line:
                    each_list = each.split(' ')
                    userFeature_dict[each_list[0]] = ' '.join(each_list[1:])

                    print(userFeature_dict[each_list[0]])
                userFeature_data.append(userFeature_dict)
                if i % 100000 == 0:
                     print(i)
            user_feature = pd.DataFrame(userFeature_data)
            user_feature.to_csv(USER_CSV, index=False)
    train=pd.read_csv(TRAIN_CSV)
    predict=pd.read_csv(TEST_CSV)

    print('predict')
    train.loc[train['label']==-1,'label']=0
    predict['label']=-1
    data=pd.concat([train,predict])
    data=pd.merge(data,ad_feature,on='aid',how='inner')
    data=pd.merge(data,user_feature,on='uid',how='inner')
    data=data.fillna('-1')
    return data

