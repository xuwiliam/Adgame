import sys
from sklearn.metrics import roc_auc_score
from random import random
import numpy as np

def eval(preds, labels, test_ads):
    dict = {}
    for i, label in enumerate(labels):
        if not dict.has_key(test_ads[i]):
            dict[test_ads[i]] = {}
            dict[test_ads[i]]['preds'] = [preds[i]]
            dict[test_ads[i]]['labels'] = [labels[i]]
        else:
            dict[test_ads[i]]['preds'].append(preds[i])
            dict[test_ads[i]]['labels'].append(labels[i])

    total = 0.0
    for key in dict:
        total += roc_auc_score(dict[key]['labels'],dict[key]['preds'])
    return total/len(dict)

path_1 = sys.argv[1]

path_2 = sys.argv[2]

path_3 = sys.argv[3]

test_ads = []

f = open(path_3)
for line in f:
    test_ads.append(int(line.strip()))
f.close()

labels = []
f = open(path_2)
for line in f:
    labels.append(float(line.strip().split(' ')[0]))
f.close()

preds = []
f = open(path_1)
for line in f:
    preds.append(float(line.strip()))
f.close()

print eval(preds, labels, test_ads)