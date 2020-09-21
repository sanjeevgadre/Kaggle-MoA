#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 16:44:17 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
#%% Get Data
X_train = pd.read_csv('./data/train_features.csv')
X_test = pd.read_csv('./data/test_features.csv')
y_train = pd.read_csv('./data/train_targets_scored.csv')

sigid_train = X_train['sig_id']
sigid_test = X_test['sig_id']

'''We know that for cp_type == ctl_vehicle there are no MoA (i.e. MoA = 0). This means that for test set records with cp_type == ctl_vehicle we already know the correct predictions and these records can be added to the train set and thereby augment the train set.
'''

X_test_ctl_idx = [i for i in X_test.index if X_test.loc[i, 'cp_type'] == 'ctl_vehicle']

X = pd.concat([X_train, X_test.iloc[X_test_ctl_idx, :]], axis = 0)
X.reset_index(drop = True, inplace = True)

y = np.array(y_train.shape[1] * len(X_test_ctl_idx) * [0])
y = y.reshape(len(X_test_ctl_idx), y_train.shape[1])
y = np.vstack((y_train, y))
y = pd.DataFrame(data = y, columns = y_train.columns)

X = X.iloc[:, 1:]
X.loc[:, ['cp_type', 'cp_time', 'cp_dose']] = X.loc[:, ['cp_type', 'cp_time', 'cp_dose']].astype('category')
y = y.iloc[:, 1:10]
y = y.astype('int')

X = pd.get_dummies(X)
X = StandardScaler().fit_transform(X)

#%% Choose an estimator method to use
est = LogisticRegression(max_iter = 1000, multi_class = 'multinomial')
est = GradientBoostingClassifier()

#%% Fit a multilabel classifier and calculate 5-fold cross validated error score

clf = OneVsRestClassifier(est)
cv_scores = cross_val_score(clf, X, y, scoring = 'neg_log_loss', cv = 2,
                            n_jobs = -1, pre_dispatch = '1.5*n_jobs',
                            error_score = 'raise')

print('Mean Cross Val Score %.6f' % np.mean(cv_scores))
