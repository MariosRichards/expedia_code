
# coding: utf-8

# In[38]:

# basic set of imports
import re, operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from itertools import chain


from IPython.display import display, HTML
import datetime, pickle
import time
import tables
get_ipython().magic('matplotlib inline')

import ml_metrics as metrics
import operator
import random

https://www.dataquest.io/blog/kaggle-tutorial/
https://www.kaggle.com/c/expedia-hotel-recommendations/forums/t/20684/a-tutorial-to-get-you-started
https://www.kaggle.com/manels/expedia-hotel-recommendations/dataquest-tutorial/run/228064

# In[39]:

filename = "destinations"
destinations = pd.read_csv(r"..\expedia_data\{0}.csv".format(filename))


# In[67]:




# In[40]:

filename = 'test_as_hdf'
test = pd.read_hdf(r"..\expedia_data\{0}.h5".format(filename))
test.info(memory_usage='deep')


# In[41]:

filename = 'train_as_hdf'
train = pd.read_hdf(r"..\expedia_data\{0}.h5".format(filename))
train.info(memory_usage='deep')


# In[42]:

import gc
gc.collect()


# In[43]:

train["hotel_cluster"].value_counts()


# In[44]:

train.shape


# In[45]:

test.shape


# In[46]:

test_ids = set(test.user_id.unique())


# In[47]:

train_ids = set(train.user_id.unique())


# In[48]:

intersection_count = len(test_ids & train_ids)
intersection_count == len(test_ids)


# In[49]:

train["year"] = train["date_time"].dt.year
train["month"] = train["date_time"].dt.month


# In[50]:

train.columns


# In[51]:

unique_users = train.user_id.unique()


# In[52]:

sel_user_ids = [unique_users[i] for i in sorted(random.sample(range(len(unique_users)), 10000)) ]


# In[53]:

sel_train = train[train.user_id.isin(sel_user_ids)]


# In[54]:

t1 = sel_train[((sel_train.year == 2013) | ((sel_train.year == 2014) & (sel_train.month < 8)))]


# In[55]:

t2 = sel_train[((sel_train.year == 2014) & (sel_train.month >= 8))]
t2 = t2[t2.is_booking == True]


# In[ ]:

### this is where you can swtich t1 with train and t2 with test


# In[ ]:

t1 = train
t2 = test


# In[ ]:




# In[56]:

most_common_clusters = list(train.hotel_cluster.value_counts().head().index)


# In[57]:

most_common_clusters


# In[58]:

predictions = [most_common_clusters for i in range(t2.shape[0])]


# In[59]:

target = [[l] for l in t2["hotel_cluster"]]
metrics.mapk(target, predictions, k=5)


# In[60]:

## Vik tutorial-maker gets 0.058020770920711007 here - quite a difference - but we *do* have a random sample of 10,000


# In[61]:

list(train.columns).index('hotel_cluster')


# In[ ]:

tr_hc = pd.DataFrame(train["hotel_cluster"])


# In[64]:

features = ['site_name', 'posa_continent', 'user_location_country',
       'user_location_region', 'user_location_city',
       'orig_destination_distance', 'user_id', 'is_mobile', 'is_package',
       'channel', 'srch_adults_cnt', 'srch_children_cnt',
       'srch_rm_cnt', 'srch_destination_id', 'srch_destination_type_id',
       'is_booking', 'cnt', 'hotel_continent', 'hotel_country', 'hotel_market',
       'year', 'month']
# WARNING V MEMORY INTENSIVE - Didn't work on 16GB win64 - still crashed with 8Gb SSD swap
# train.corr()["hotel_cluster"]
for ft in features:
    print(ft,tr_hc.corrwith(train[ft]))


# In[65]:

from sklearn.decomposition import PCA

pca = PCA(n_components=3)
dest_small = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
dest_small = pd.DataFrame(dest_small)
dest_small["srch_destination_id"] = destinations["srch_destination_id"]



# In[66]:

dest_small


# In[68]:

def calc_fast_features(df):
#     df["date_time"] = pd.to_datetime(df["date_time"])
#     df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce")
#     df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce")
    
    props = {}
    for prop in ["month", "day", "hour", "minute", "dayofweek", "quarter"]:
        props[prop] = getattr(df["date_time"].dt, prop)
    
    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        props[prop] = df[prop]
    
    date_props = ["month", "day", "dayofweek", "quarter"]
    for prop in date_props:
        props["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        props["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    props["stay_span"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')
        
    ret = pd.DataFrame(props)
    
    ret = ret.join(dest_small, on="srch_destination_id", how='left', rsuffix="dest")
    ret = ret.drop("srch_destination_iddest", axis=1)
    return ret



# In[69]:

df = calc_fast_features(t1)
df.fillna(-1, inplace=True)


# In[70]:

df


# In[71]:

# Random forest, classifying to 100 categories
predictors = [c for c in df.columns if c not in ["hotel_cluster"]]
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
scores = cross_validation.cross_val_score(clf, df[predictors], df['hotel_cluster'], cv=3)
scores


# In[90]:

# Random forest, but with only binary classification
all_probs = []
unique_clusters = df["hotel_cluster"].unique()


# In[ ]:




# In[ ]:




# In[91]:

for cluster in unique_clusters:
    df["target"] = 1
    #df[df['A'] > 2]['B'] = new_val
    #df.loc[df['A'] > 2, 'B'] = new_val
    df.loc[df["hotel_cluster"] != cluster, "target"] = 0
#     df["target"][df["hotel_cluster"] != cluster] = 0
    predictors = [col for col in df if col not in ['hotel_cluster', "target"]]
    probs = []
    cv = KFold(len(df["target"]), n_folds=2)
    clf = RandomForestClassifier(n_estimators=10, min_weight_fraction_leaf=0.1)
    for i, (tr, te) in enumerate(cv):
        clf.fit(df[predictors].iloc[tr], df["target"].iloc[tr])
        preds = clf.predict_proba(df[predictors].iloc[te])
        probs.append([p[1] for p in preds])
    full_probs = chain.from_iterable(probs)
    all_probs.append(list(full_probs))



# In[93]:

prediction_frame = pd.DataFrame(all_probs).T
prediction_frame.columns = unique_clusters
def find_top_5(row):
    return list(row.nlargest(5).index)

preds = []
for index, row in prediction_frame.iterrows():
    preds.append(find_top_5(row))



# In[98]:




# In[100]:

metrics.mapk([[l] for l in t2["hotel_cluster"]], preds, k=5)


# In[101]:

def make_key(items):
    return "_".join([str(i) for i in items])

match_cols = ["srch_destination_id"]
cluster_cols = match_cols + ['hotel_cluster']
groups = t1.groupby(cluster_cols)
top_clusters = {}
for name, group in groups:
    clicks = len(group.is_booking[group.is_booking == False])
    bookings = len(group.is_booking[group.is_booking == True])
    
    score = bookings + .15 * clicks
    
    clus_name = make_key(name[:len(match_cols)])
    if clus_name not in top_clusters:
        top_clusters[clus_name] = {}
    top_clusters[clus_name][name[-1]] = score


# In[102]:




# In[103]:


cluster_dict = {}
for n in top_clusters:
    tc = top_clusters[n]
    top = [l[0] for l in sorted(tc.items(), key=operator.itemgetter(1), reverse=True)[:5]]
    cluster_dict[n] = top


# In[104]:

preds = []
for index, row in t2.iterrows():
    key = make_key([row[m] for m in match_cols])
    if key in cluster_dict:
        preds.append(cluster_dict[key])
    else:
        preds.append([])


# In[105]:

metrics.mapk([[l] for l in t2["hotel_cluster"]], preds, k=5)


# In[106]:

match_cols = ['user_location_country', 'user_location_region', 'user_location_city', 'hotel_market', 'orig_destination_distance']

groups = t1.groupby(match_cols)
    
def generate_exact_matches(row, match_cols):
    index = tuple([row[t] for t in match_cols])
    try:
        group = groups.get_group(index)
    except Exception:
        return []
    clus = list(set(group.hotel_cluster))
    return clus

exact_matches = []
for i in range(t2.shape[0]):
    exact_matches.append(generate_exact_matches(t2.iloc[i], match_cols))


# In[108]:

def f5(seq, idfun=None): 
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result
    
full_preds = [f5(exact_matches[p] + preds[p] + most_common_clusters)[:5] for p in range(len(preds))]
metrics.mapk([[l] for l in t2["hotel_cluster"]], full_preds, k=5)


# In[ ]:




# In[109]:

write_p = [" ".join([str(l) for l in p]) for p in full_preds]
write_frame = ["{0},{1}".format(t2["id"][i], write_p[i]) for i in range(len(full_preds))]
write_frame = ["id,hotel_clusters"] + write_frame
with open("predictions.csv", "w+") as f:
    f.write("\n".join(write_frame))


# In[ ]:



