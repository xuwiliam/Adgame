import pandas as pd
train_data = pd.read_csv('../../data/final_competition/combine_train.csv',sep=',')
df = [0]*5
columns = train_data.columns
df[0] = train_data.loc[0:9107939,['label','uid']]
df[1] = train_data.loc[9107940:18215879,['label','uid']]
df[2] = train_data.loc[18215880:27323819,['label','uid']]
df[3] = train_data.loc[27323820:36431759,['label','uid']]
df[4] = train_data.loc[36431760:45539699,['label','uid']]

df[0].to_csv('spilt_train_1.csv',sep=',')
df[1].to_csv('split_train_2.csv',sep=',')
df[2].to_csv('split_train_3.csv',sep=',')
df[3].to_csv('split_train_4.csv',sep=',')
df[4].to_csv('split_train_5.csv',sep=',')

def StaticUserRatio(data,part_index):
    labels = []
    uids = []
    uid_dict={}
    for df in data:
        labels.extend(df['label'])
        uids.extend(df['uid'])

    f = open('data_part'+str(part_index),'w+')
    for i,id in enumerate(uids):
        if id not in uid_dict:
            uid_dict[id]={}
            uid_dict[id]['1']=0
            uid_dict[id]['all']=0
        uid_dict[id]['all']+=1
        if str(labels[i])=='1':
           uid_dict[id]['1']+=1
    for key in uid_dict:
        res = uid_dict[key]['1']*1.0/uid_dict[key]['all']
        f.write(str(key)+','+str(res)+'\n')



if __name__ == '__main__':
   df=[0]*5
   df[0] = pd.read_csv('split_train_1.csv',sep=',')
   df[1] = pd.read_csv('split_train_2.csv',sep=',')
   df[2] = pd.read_csv('split_train_3.csv',sep=',')
   df[3] = pd.read_csv('split_train_4.csv',sep=',')
   df[4] = pd.read_csv('split_train_5.csv',sep=',')
   cnt = 0
   for i in range(0,5):
       sample_data = []
       for j in range(0,5):
           if j%5!=cnt:
              sample_data.append(df[j])

       StaticUserRatio(sample_data,cnt)
       cnt+=1
       sample_data=None




