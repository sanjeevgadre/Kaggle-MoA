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

