# -*- coding: utf-8 -*-

#from kdd2017.models import *
#
#
#def remove_outliers_by_classifier(X, y, dates, ):
#    xgboost = XGBoost(max_depth=2)
#    xgboost.fit(X, y)
#    y_pred = xgboost.predict(X)
#    diff_values = np.abs(y_pred - y)
#    abs_diff_vals = np.abs(diff_values)
#    sorted_indexes = sorted(range(len(abs_diff_vals)), key = lambda x: abs_diff_vals[x])
#    sorted_indexes_lead = sorted_indexes[:int(len(abs_diff_vals)*0.9)]
#    return X[sorted_indexes_lead], y[sorted_indexes_lead], dates[sorted_indexes_lead]
