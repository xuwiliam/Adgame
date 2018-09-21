import pandas as pd


def combine(data, left_feature, right_feature):
    feature_combination = {}

    # traverse all feature
    for left in left_feature:
        for right in right_feature:
            # build key
            key = left + '|' + right
            if key in data.columns:
                print(key + 'is existed')
            print('key',key)
            feature_combination[key] = []

            for i in range(len(data)):
                idata = data.iloc[i]
                left_vals = str(idata[left]).split(' ')
                right_vals = str(idata[right]).split(' ')

                # combine
                combination = []
                for lv in left_vals:
                    for rv in right_vals:
                        val = lv + '|' + rv
                        combination.append(val)
                feature_combination[key].append(' '.join(combination))
                #print('combination: ',combination)
    # add in data
    for key in feature_combination:
        data[key] = pd.Series(feature_combination[key], index=data.index)
    print('data columns',data.columns)
    return data
