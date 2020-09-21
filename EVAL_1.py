#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:44:17 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

#%% Get Data
X = pd.read_csv('./data/train_features.csv')
y = pd.read_csv('./data/train_targets_scored.csv')

X = X.iloc[:, 1:]
X.loc[:, ['cp_type', 'cp_time', 'cp_dose']] = X.loc[:, ['cp_type', 'cp_time', 'cp_dose']].astype('category')
y = y.iloc[:, 1:10]

X = pd.get_dummies(X)
X = StandardScaler().fit_transform(X)

#%% Choose an estimator method to use
est = LogisticRegression(max_iter = 1000, multi_class = 'multinomial')

#%% Fit a multilabel classifier and calculate 5-fold cross validated error score

clf = OneVsRestClassifier(est)
cv_scores = cross_val_score(clf, X, y, scoring = 'neg_log_loss',
                            n_jobs = -1, pre_dispatch = '1.5*n_jobs')

print(np.mean(cv_scores))