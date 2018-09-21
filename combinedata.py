import pandas as pd
df_ad = pd.read_csv('adFeature.csv',sep=',')
df_train = pd.read_csv('train.csv',sep=',')
dict_user={}
user_headers = ['uid','age','gender','marriageStatus','education','consumptionAbility','LBS','interest1','interest2','interest3',
                'interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','appIdInstall','appIdAction','ct','os',
                'carrier','house']
for key in user_headers:
    dict_user[key] = []
def genUserTable():
    f = open('userFeature.data','r')
    lines  = f.readlines()
    f.seek(0)
    hash_user={}

    for line in lines:
        for feature in user_headers:
                hash_user[feature] = 0
        data = line.strip().split('|')
        content = {}
        for item in data:
            detail = item.split(' ')
            content[detail[0]] = ' '.join(str(d) for d in detail[1:-1])
        for key in dict_user:
            if key in content:
               dict_user[key].append(content[key])
            else:
               dict_user[key].append('-1')
    df_user = pd.DataFrame(dict_user,columns=user_headers)
    df_user.to_csv('user_feature.csv',sep=',')
    f.close()
def combinedata(ad,user,data,data_type):
    if data_type == 'train':
       combinetrain = data.join([ad,user],on=['aid','uid'],how='left')
       combinetrain.fillna('-1')
       combinetrain.to_csv('combine_train.txt',sep=',',index=False)
    if data_type == 'test1':
        combinetest= data.join([ad, user], on=['aid', 'uid'], how='left')
        combinetest.fillna('-1')
        combinetest.to_csv('combine_test1.txt', sep=',', index=False)
    if data_type == 'test2':
        combinetest= data.join([ad, user], on=['aid', 'uid'], how='left')
        combinetest.fillna('-1')
        combinetest.to_csv('combine_test2.txt', sep=',', index=False)
if __name__ == '__main__':
   genUserTable()