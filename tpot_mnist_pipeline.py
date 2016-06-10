import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR')
training_indices, testing_indices = train_test_split(tpot_data.index, stratify = tpot_data['class'].values, train_size=0.75, test_size=0.25)


result1 = tpot_data.copy()

result1 = tpot_data.copy()

# Combine two DataFrames
result1 = result1.join(result1[[column for column in result1.columns.values if column not in result1.columns.values]])

# Use Scikit-learn's PolynomialFeatures to construct new features from the existing feature set
training_features = result1.loc[training_indices].drop('class', axis=1)

if len(training_features.columns.values) > 0 and len(training_features.columns.values) <= 700:
    # The feature constructor must be fit on only the training data
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly.fit(training_features.values.astype(np.float64))
    constructed_features = poly.transform(result1.drop('class', axis=1).values.astype(np.float64))
    result2 = pd.DataFrame(data=constructed_features)
    result2['class'] = result1['class'].values
else:
    result2 = result1.copy()

# Perform classification with a logistic regression classifier
lrc3 = LogisticRegression(C=0.48148148148148145)
lrc3.fit(result2.loc[training_indices].drop('class', axis=1).values, result2.loc[training_indices, 'class'].values)
result3 = result2.copy()
result3['lrc3-classification'] = lrc3.predict(result3.drop('class', axis=1).values)

# Use Scikit-learn's Recursive Feature Elimination (RFE) for feature selection
training_features = result3.loc[training_indices].drop('class', axis=1)
training_class_vals = result3.loc[training_indices, 'class'].values

if len(training_features.columns.values) == 0:
    result4 = result3.copy()
else:
    selector = RFE(SVC(kernel='linear'), n_features_to_select=min(77, len(training_features.columns)), step=0.99)
    selector.fit(training_features.values, training_class_vals)
    mask = selector.get_support(True)
    mask_cols = list(training_features.iloc[:, mask].columns) + ['class']
    result4 = result3[mask_cols]
