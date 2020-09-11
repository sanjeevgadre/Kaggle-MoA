#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 14:51:03 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

#%% Get data
x_train = pd.read_csv('./data/train_features.csv')
x_test = pd.read_csv('./data/test_features.csv')
y_train_scored = pd.read_csv('./data/train_targets_scored.csv')
y_train_unscored = pd.read_csv('./data/train_targets_nonscored.csv')

#%% 1
print('The train dataset has %i records and %i features' % (x_train.shape[0], x_train.shape[1]))
# 23814 rows and 876 features

print('The test dataset has %i records and %i features' % (x_test.shape[0], x_test.shape[1]))
# 3982 rows and 876 features

print('The train_target_scored dataset has %i records and %i categories' 
      % (y_train_scored.shape[0], y_train_scored.shape[1]))
# 23814 rows and 207 categories

print('The train_target_unscored dataset has %i records and %i categories'
      % (y_train_unscored.shape[0], y_train_unscored.shape[1]))
# 23814 rows and 403 categories

print('List of features that differ across train and test datasets ', 
      x_train.columns.difference(x_test.columns))
# Empty. Both train and test datasets have the same columns

print('List of categories that are in both scored_train_target and unscored_train_target datasets')
print(y_train_scored.columns.intersection(y_train_unscored.columns))
# Apart from sig_id, there are no common categories across scored_train_target and unscored_train_target

print('Sample rows of train dataset\n', x_train.head())
print('Sample rows of test dataset\n', x_test.head())
print('Sample rows of train_target_scored dataset\n', y_train_scored.head())
print('Sample rows of train_target_unscored dataset\n', y_train_unscored.head())

print('Features with missing values in train dataset')
print([c for c in x_train.columns if x_train.isnull().sum()[c] != 0])
# Empty

print('Features with missing values in test dataset')
print([c for c in x_test.columns if x_test.isnull().sum()[c] != 0])
# Empty

#%% 2
# Features['cp_type', 'cp_time', 'cp_dose'] are categorical while the rest are numerical
# Comparing levels of categorical features across train and test dataset
cat_feats = ['cp_type', 'cp_time', 'cp_dose']
for feat in cat_feats:
    print('For feature %s' % feat)
    print('Levels in train dataset --> %s' % x_train[feat].unique())
    print('Levels in test dataset --> %s' % x_test[feat].unique())
# For the three categorical levels both train and test dataset have the same number of levels

# Comparing distribution of records across levels of categorical features for train and test dataset
for feat in cat_feats:
    print('Distribution of records across levels of %s for train and test dataset' % feat)
    print(pd.DataFrame(data = {'Train' : x_train[feat].value_counts(normalize = True), 
                               'Test' : x_test[feat].value_counts(normalize = True)}))
    print('')
# The distribution of records across levels of categorical features for train and test dataset are comparable

# Comparing if a non-categorical feature across the train and test dataset come from the same distribution. We use the Kolmogorov-Smirnov test to test the hypothesis
ks_pvalues = []
for feat in x_train.columns:
    if feat in cat_feats or feat == 'sig_id':
        ks_pvalues.append(None)
    else:
        _, pvalue = stats.ks_2samp(x_train[feat], x_test[feat])
        ks_pvalues.append(pvalue)

diff_pop_cols = []
for pvalue in ks_pvalues:
    if pvalue is None or pvalue > 0.01:
        continue
    else:
        diff_pop_cols.append(x_train.columns[ks_pvalues.index(pvalue)])

print('Using the Kolmogorov-Smirnov test and a threshold p-value = 0.05,\nfor the following features, we reject the hypothesis that the values\nin the train and test datasets come from the same population')
print(np.array(diff_pop_cols))
# There are 152 columns where we can't be certain that the train and test samples come from the same population
'''
['g-0' 'g-3' 'g-11' 'g-22' 'g-37' 'g-52' 'g-60' 'g-63' 'g-68' 'g-72'
 'g-75' 'g-76' 'g-82' 'g-86' 'g-98' 'g-100' 'g-101' 'g-102' 'g-112'
 'g-134' 'g-139' 'g-140' 'g-142' 'g-145' 'g-149' 'g-153' 'g-154' 'g-157'
 'g-158' 'g-163' 'g-167' 'g-169' 'g-173' 'g-175' 'g-177' 'g-181' 'g-186'
 'g-191' 'g-192' 'g-193' 'g-195' 'g-197' 'g-198' 'g-203' 'g-204' 'g-206'
 'g-211' 'g-215' 'g-216' 'g-224' 'g-228' 'g-235' 'g-236' 'g-237' 'g-239'
 'g-248' 'g-250' 'g-255' 'g-257' 'g-267' 'g-298' 'g-300' 'g-305' 'g-321'
 'g-322' 'g-330' 'g-332' 'g-335' 'g-354' 'g-361' 'g-365' 'g-380' 'g-384'
 'g-386' 'g-392' 'g-400' 'g-406' 'g-410' 'g-413' 'g-416' 'g-418' 'g-437'
 'g-439' 'g-440' 'g-450' 'g-460' 'g-467' 'g-485' 'g-486' 'g-489' 'g-503'
 'g-506' 'g-517' 'g-518' 'g-526' 'g-554' 'g-567' 'g-568' 'g-574' 'g-577'
 'g-590' 'g-598' 'g-600' 'g-618' 'g-619' 'g-628' 'g-639' 'g-664' 'g-666'
 'g-670' 'g-675' 'g-677' 'g-687' 'g-696' 'g-700' 'g-702' 'g-721' 'g-723'
 'g-731' 'g-736' 'g-744' 'g-745' 'g-757' 'g-768' 'g-770' 'c-4' 'c-5' 'c-6'
 'c-9' 'c-10' 'c-21' 'c-22' 'c-24' 'c-27' 'c-36' 'c-47' 'c-52' 'c-54'
 'c-55' 'c-59' 'c-60' 'c-62' 'c-65' 'c-70' 'c-72' 'c-75' 'c-77' 'c-79'
 'c-86' 'c-87' 'c-93' 'c-96']
'''

#%% 3 - For the columns that do indeed come from the same population, does PCA reduce the dimensionality

# Combining train and test datasets including only those columns that likely come from the same population
cols_to_ignore = cat_feats + diff_pop_cols
cols_to_ignore.append('sig_id')
df = pd.concat([
    x_train.loc[:, [x for x in x_train.columns if x not in cols_to_ignore]],
    x_test.loc[:, [x for x in x_test.columns if x not in cols_to_ignore]]
    ], axis = 0)

pca = PCA()
model = pca.fit(df)
exp_var_ratio = model.explained_variance_ratio_
exp_var_ratio = np.cumsum(exp_var_ratio)

print('95%% of the variance is explained by the first %i principal components' 
      % len(exp_var_ratio[exp_var_ratio <= 0.95]))
# 470

