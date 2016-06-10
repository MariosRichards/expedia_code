# performance notes
# 1800Mb base -> 
# 500000 0.785978 130.03799221073106, 11Gb
# 500000 0.786076 129.72322886959992
# 500000 0.785976 129.8444541967028

# 600000 0.786488333333 159.2348512115112
# now with a stride of 5
# 600000 0.778443333333 163.22313340061095




import numpy as np
import pandas as pd

import time

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier



# NOTE: Make sure that the class is labeled 'class' in the data file

filename = 'train_bookings_only'
train = pd.read_hdf(r"..\expedia_data\{0}.h5".format(filename))

drop_list = [ "cnt","ci_year", "co_year", "ci_month", "co_month", "ci_day", "co_day", "stay_span", "booking_span" ,"orig_destination_distance", "user_id", "srch_destination_id", "is_booking"]
#df=df.rename(columns = {'two':'new_name'})
train = train.rename(columns = {'hotel_cluster':'class'}).drop(drop_list,axis=1)
train.columns
print("train")
train.info(memory_usage='deep')

filename = 'test_as_hdf'
test = pd.read_hdf(r"..\expedia_data\{0}.h5".format(filename))

drop_list = [ "ci_year", "co_year", "ci_month", "co_month", "ci_day", "co_day", "stay_span", "booking_span" ,"orig_destination_distance", "user_id", "srch_destination_id"]
#df=df.rename(columns = {'two':'new_name'})
test = test.drop(drop_list,axis=1)
test.columns
print("test")
test.info(memory_usage='deep')

# some issues with things that don't convert to float16!


# taking a slice off the top (terrible approach!)
# train=train[0:600000]
train=train[0:3000000:5]

tpot_data = train
tpot_data.columns

import gc
gc.collect()

training_indices, testing_indices = train_test_split(tpot_data.index,
                                                     stratify = tpot_data['class'].values,
                                                     train_size=0.75,
                                                     test_size=0.25)
                                                     
result1 = tpot_data.copy()

start = time.clock()
rfc1 = RandomForestClassifier(n_estimators=17, max_features=min(62, len(result1.columns) - 1))
rfc1.fit(result1.loc[training_indices].drop('class', axis=1).values, result1.loc[training_indices, 'class'].values)
result1['rfc1-classification'] = rfc1.predict(result1.drop('class', axis=1).values)

end = time.clock()
duration = end - start
print(train.shape[0],sum(result1['rfc1-classification']==result1['class'])/result1.shape[0], duration)

del train, result1
import gc
gc.collect()


test['rfc1-classification'] = rfc1.predict(test.drop('id', axis=1).values)


"id",


submission = pd.DataFrame({
        "id": test["id"],
        "hotel_cluster": test["rfc1-classification"]
    })
    
submission.to_csv("expedia.csv", index=False)  
