import argparse, csv, sys, pickle, collections, math

def logistic_func(x):
    return 1/(1+math.exp(-x))

def inv_logistic_func(x):
    return math.log(x/(1-x))

path_1 = sys.argv[1]
path_2 = sys.argv[2]
path_3 = sys.argv[3]

data_1 = []
data_2 = []

f = open(path_1,'rb')
f.readline()
for line in f:
    data_1.append(float(line.strip().split(',')[-1]))
f.close()

f = open(path_2,'rb')
for line in f:
    data_2.append(float(line))
f.close()

assert len(data_1) == len(data_2)

f = open(path_3,'wb')
for i, d in enumerate(data_1):
    t1 = inv_logistic_func(d)
    t2 = inv_logistic_func(data_2[i])
    val = logistic_func((t1+t2)/2)
    f.write(str(val)+'\n')
f.close()