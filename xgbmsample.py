#! /usr/bin/python
import numpy as np
import xgboost as xgb

from numpy import genfromtxt

# label need to be 0 to num_class -1
data = np.genfromtxt('subtrain.csv', delimiter=',')
datest = np.genfromtxt('test.csv', delimiter=',')

#data = np.loadtxt('./dermatology.data', delimiter=',',converters={33: lambda x:int(x == '?'), 34: lambda x:int(x)-1 } )
sz = data.shape

def map5eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(1,6)]
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', metric

traincol=np.array([2,3,4,5,6,8,9,10,11,12,14,15,16,17,18,21,22])
testcol=np.array([2,3,4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])

train=data[:,traincol]
test=datest[1:,testcol]

#train = data[:int(sz[0] * 0.7), :]
#test = data[int(sz[0] * 0.7):, :]

train_Y = data[:,23]
test_Y = datest[1:,2]*0
#train_Y = train[:, 34]


xg_train = xgb.DMatrix( train, label=train_Y)
xg_test = xgb.DMatrix(test, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
#param['eval_metric'] = 'map5eval'
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 6

def apk(actual, predicted, k=5):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk5(actual, predicted, k=5):
    #print (predicted)
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

params = {}
params["objective"] = "multi:softprob"
params["booster"] = "gbtree"
params["num_class"] = 100
params["eta"] = 0.01
params["subsample"] = 0.75
params["colsample_bytree"] = 0.75
params["max_depth"] = 6
params["min_child_weight"] = 3
params["silent"] = 1
xgtrain = xgb.DMatrix(train, train_Y)
print(xgb.cv(params, xgtrain, 1000, nfold=5, feval= mapk5, early_stopping_rounds=50))