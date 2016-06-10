import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import VarianceThreshold

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)


result1 = tpot_data.copy()

# Use Scikit-learn's VarianceThreshold for feature selection
training_features = result1.loc[training_indices].drop('class', axis=1)

selector = VarianceThreshold(threshold=0.01)
try:
    selector.fit(training_features.values)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    result1 = result1[mask_cols]
except ValueError:
    # None of the features meet the variance threshold
    result1 = result1[['class']]
