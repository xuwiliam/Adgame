import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
import featureCombinator
def build_feature(data):
      one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId','adCategoryId', 'productId', 'productType']
      vector_feature=['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
      for feature in one_hot_feature:
          try:
              data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
          except:
              data[feature] = LabelEncoder().fit_transform(data[feature])


      # 特征组合
      left_feature = ['aid']
      right_feature = ['interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
      #  right_feature = ['interest5']
      data = featureCombinator.combine(data, left_feature, right_feature)

      train=data[data.label!=-1]
      train_y=train.pop('label')
      # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
      test=data[data.label==-1]
      res=test[['aid','uid']]
      print('res',res)
      print(res['uid'].values)
      test=test.drop('label',axis=1)
      enc = OneHotEncoder()
      train_x=train[['creativeSize']]
      test_x=test[['creativeSize']]


      # one-hot
      for feature in one_hot_feature:
          print(feature)
          enc.fit(data[feature].values.reshape(-1, 1))
          train_a = enc.transform(train[feature].values.reshape(-1, 1))
          test_a = enc.transform(test[feature].values.reshape(-1, 1))
          print('train_x','train_a',train_x,train_a)
          train_x= sparse.hstack((train_x, train_a))
          print('hstack_train_x')
          print(train_x)
          test_x = sparse.hstack((test_x, test_a))
          print('hstack_text_x')
          print(test_x)

      '''
      # TF
      cv=CountVectorizer()
      for feature in vector_feature:
           print feature
           cv.fit(data[feature])
           train_a = cv.transform(train[feature])
           test_a = cv.transform(test[feature])
           train_x = sparse.hstack((train_x, train_a))
           test_x = sparse.hstack((test_x, test_a))
      print('cv prepared !')
      '''
      '''
      # tf-idf
      tv = TfidfVectorizer()
      for feature in vector_feature:
           print feature
           tv.fit(data[feature])
           tran_a = tv.transform(train[feature])
           test_a = cv.transform(test[feature])
           train_x = sparse.hstack((train_x, train_a))
           test_x = sparse.hstack((test_x, test_a))
      print('tf-idf prepared')
      '''

      #aid+x双特征组合
      cv=CountVectorizer()
      for left in left_feature:
         for right in right_feature:
             feature = left+'|'+right
             print(feature)
         cv.fit(data[feature])
         train_a = cv.transform(train[feature])
         test_a = cv.transform(test[feature])
         train_x = sparse.hstack((train_x, train_a))
         test_x = sparse.hstack((test_x, test_a))
      print('feature combination cv prepared !')


      return train_x, train_y, test_x, res