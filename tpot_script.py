import os
mingw_path = 'C:\\Program Files\\mingw-w64\\x86_6~4-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

import re, operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from collections import defaultdict
from IPython.display import display, HTML
import datetime, pickle
import time
import tables
import csv

from tpot import TPOT

filename = 'train_bookings_only'
train = pd.read_hdf(r"..\expedia_data\{0}.h5".format(filename))


drop_list = [ "ci_year", "co_year", "ci_month", "co_month", "ci_day", "co_day", "stay_span", "booking_span" ,"orig_destination_distance", "user_id", "srch_destination_id", "is_booking"]
train = train.drop(drop_list,axis=1)
train = train[0:3000000:300]
train.info(memory_usage='deep')



X = train.drop("hotel_cluster",axis=1).values
y = train.loc[: , "hotel_cluster"].values

del train
import gc
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,test_size=0.25)

print("got here!")

my_tpot = TPOT(generations=20,verbosity=2,population_size=5) # seems to have a problem with pop <5
# gen 1-> really means two generations!

start = time.clock()
print(start)
my_tpot.fit(X_train, y_train)
my_tpot.export('tpot_expedia_pipeline.py')
end = time.clock()
duration = end - start
score = my_tpot.score(X_test, y_test)
print(duration,score)


 