# -*- coding: utf-8 -*-
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
import inspect
import numpy as np
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta
from datetime import datetime
import numpy as np
from kdd2017.utils import invboxcox
from kdd2017.utils import mape_loss
from kdd2017.utils import remove_outliers
from kdd2017.utils import remove_outliers2
from kdd2017.utils import remove_outliers3
from kdd2017.utils import invboxcox
from kdd2017.utils import compute_harmonic_mean
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
import collections
import xgboost as xgb
from scipy.stats import boxcox
from sklearn.decomposition import PCA
import copy
from random import shuffle
import lightgbm as lgb
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import os
import json
from time import sleep
import random
from sklearn.model_selection import train_test_split

global_variables = {}


random_state = 1


def mspe(y, dtrain):
    yhat = dtrain.get_label()
    grad = 2.0/yhat * (y * 1.0 / yhat - 1)
    hess = 2.0/(yhat**2)
    return grad, hess

def L2(pred,true):
    loss = np.square(pred-true)
    return loss.mean()

def L1(pred,true):
    loss = np.abs(pred-true)
    return loss.mean()

def SMAPE(pred,true):
    loss = np.abs((pred-true)/(pred+true))
    return loss.mean()

#This function chooses the best point estimate for a numpy array, according to a particular loss.
#The loss function should take two numpy arrays as arguments, and return a scalar. One example is SMAPE, see above.
def solver(x,loss):
    mean = x.mean()
    best = loss(mean,x)
    result = mean
    for i in x:
        score = loss(i,x)
        if score < best:
            best = score
            result = i
    return result

class NonparametricKNN(object):
    def __init__(self,n_neighbors=5,loss='L2'):
        if loss in ['L1','L2','SMAPE']:
            loss = {'L1':L1,'L2':L2,'SMAPE':SMAPE}[loss]
        self.loss = loss
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors,algorithm='auto',n_jobs=-1)
        self.solver = lambda x:solver(x,loss)
    def __repr__(self, ):
        return "NonparametricKNN: loss:" + repr(self.loss) + ", n_neighbors=" + repr(self.n_neighbors)
    def __str__(self,):
        return repr(self)

    def fit(self,train,target):#All inputs should be numpy arrays.
        self.model.fit(train)
        self.f=np.vectorize(lambda x:target[x])
        return self

    def predict(self,test):#Return predictions as a numpy array.
        neighbors = self.model.kneighbors(test,return_distance=False)
        neighbors = self.f(neighbors)
        result = np.apply_along_axis(self.solver,1,neighbors)
        return result


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', mape_loss(preds, labels)

def xgboostobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess

class Pipeline(object):
    def __init__(self, **kwargs):
        self.models = kwargs["models"]
        kwargs.pop("models")

    def fit(self, X, y):
        last_y = np.copy(y)
        for i, model in enumerate(self.models):
            self.models[i].fit(X, last_y)
            last_y = np.copy(self.models[i].predict(X))

    def predict(self, X, **kwargs):
        return self.models[-1].predict(X)        

class MedianModel(object):
    def __init__(self, **kwargs):
        self.ft_pos = kwargs.get("ft_pos", np.asarray([0,1]))
        if "ft_pos" in kwargs:
            kwargs.pop("ft_pos")
    def __str__(self,):
        return "MedianModel:\n  " + repr(self.ft_pos)
    def __repr__(self,):
        return str(self)

    def fit(self, X, y, **kwargs):
        X = X[:,self.ft_pos]
        self.values = {}
        for i, x in enumerate(X):
            key = tuple([j for j in x])
            if key not in self.values:
                self.values[key] = []
            self.values[key].append(y[i])

    def predict(self, X, **kwargs):
        X = X[:,self.ft_pos]
        y = []
        for i, x in enumerate(X):
            key = tuple([j for j in x])
            y.append(np.median(self.values[key]))
        return np.asarray(y)


class XGBoost(object):
    def __init__(self, **kwargs):
        #self.eval_metric = kwargs.get("eval_metric", "logloss")
        #self.eta = kwargs.get("eta", 0.02)
        #self.max_depth = kwargs.get("max_depth", 3)
        #self.objective = kwargs.get("objective", "reg:gamma")
        #self.booster = kwargs.get("booster", "gbtree")
        self.use_mspe = kwargs.get("use_mspe", False)
        self.num_round = kwargs.get("num_round", 1500)
        self.early_stopping_rounds = kwargs.get("early_stopping_rounds", 10)
        self.verbose_eval = kwargs.get("verbose_eval", 500)
        self.eval_metric = kwargs.get("eval_metric", None)
        self.feval = None
        if self.eval_metric == "mape":
            self.eval_metric = None
            self.feval = evalerror
        if "use_mspe" in kwargs:
            kwargs.pop("use_mspe")
        if "early_stopping_rounds" in kwargs:
            kwargs.pop("early_stopping_rounds")
        if "num_round" in kwargs:
            kwargs.pop("num_round")
        if "verbose_eval" in kwargs:
            kwargs.pop("verbose_eval")
        if "eval_metric" in kwargs:
            kwargs.pop("eval_metric")
        self.param = kwargs

    def __str__(self, ):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        ret_val = ""
        for attr in members:
            value = getattr(self, attr)
            ret_val += "    (%s:%s)\n" % (attr, repr(value))
        return "XGBoost: " + ret_val

    def __repr__(self, ):
        return str(self)

    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is not None:
            print("use sample_weight")
            dtrain = xgb.DMatrix(X, label=y, weight=sample_weight, silent=True)
        else:
            dtrain = xgb.DMatrix(X, label=y, silent=True)
        evallist = [(dtrain, 'train')]
        param = {
                 'n_estimators':200,
                 'booster': 'gbtree',
                 'nthread': -1,
                 'max_depth': 3,
                 'eta': 0.02,
                 'silent': 1,
                 'objective': 'reg:gamma',
                 'colsample_bytree': 0.7,
                 'eval_metric': 'logloss',
                 'subsample': 0.5}
        if self.eval_metric is not None:
            param["eval_metric"] = self.eval_metric
        param.update(self.param)
        if not self.use_mspe:
            if self.early_stopping_rounds > 0:
                self.bst = xgb.train(param, dtrain, self.num_round,
                                     evallist, feval=self.feval, early_stopping_rounds=self.early_stopping_rounds,
                                     verbose_eval=self.verbose_eval)
            else:
                self.bst = xgb.train(param, dtrain, self.num_round,
                                     evallist, feval=self.feval, verbose_eval=self.verbose_eval)
        else:
            param.pop("objective")
            #param.pop("eval_metric")
            if self.early_stopping_rounds > 0:
                self.bst = xgb.train(param, dtrain, self.num_round,
                                     evallist,
                                     mspe, feval=self.feval, early_stopping_rounds=self.early_stopping_rounds,
                                     verbose_eval=self.verbose_eval)
            else:
                self.bst = xgb.train(param, dtrain, self.num_round,
                                     evallist,
                                     mspe, feval=self.feval,
                                     verbose_eval=self.verbose_eval)

    def predict(self, X, **kwargs):
        dtest = xgb.DMatrix(X)
        return self.bst.predict(dtest)


def evalerror_lgbm(preds, dtrain):
    labels = dtrain.get_label()
    # return a pair metric_name, result
    # since preds are margin(before logistic transformation, cutoff at 0)
    return 'error', mape_loss(preds, labels), False


class LGBM(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        #self.kwargs["random_state"] = random_state
        self.use_mspe = kwargs.get("use_mspe", False)
        if "use_mspe" in self.kwargs:
            self.kwargs.pop("use_mspe")
        self.gbm = lgb.LGBMRegressor(**self.kwargs)

    def __repr__(self,):
        return "LGBM:" + repr(self.kwargs)

    def __str__(self, ):
        return repr(self)

    def fit(self, X, y):
        if self.use_mspe:
            lgb_train = lgb.Dataset(X, y,
                        weight=np.ones(X.shape[0]), 
                        free_raw_data=False)
            lgb_test = lgb.Dataset(X, y, reference=lgb_train,
                        weight=np.ones(X.shape[0]), 
                        free_raw_data=False)
            self.gbm = lgb.train(
                self.kwargs,
                lgb_train,
                num_boost_round=10,
                fobj=mspe,
                feval=evalerror_lgbm,
                valid_sets=lgb_test)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3)
            #lgb_test = lgb.Dataset(X, y, reference=lgb_train,
            #            weight=np.ones(X.shape[0]), 
            #            free_raw_data=False) 
            self.gbm.fit(X, y, early_stopping_rounds=10, eval_set=[(X, y)], verbose=False)
            #print "gbm best_iteration=", self.gbm.best_iteration

    def predict(self, X):
        if self.use_mspe:
            return self.gbm.predict(X)
        else:
            return self.gbm.predict(X, num_iteration=self.gbm.best_iteration)

def remove_outliers_by_classifier(X, y, dates, model, m=0.9):
    #xgboost = XGBoost(max_depth=2, num_round=6000)
    if np.isnan(X).any():
        print("X contains NaN")
    if np.isinf(X).any():
        print("X contains inf")
    if np.isnan(np.log(y)).any():
        print("y contains nan")
    if np.isinf(np.log(y)).any():
        print("y contains inf")
    print("X=", X.shape)
    print("y=", y.shape)
    model.fit(X, y)
    y_pred = model.predict(X)
    diff_values = np.abs(y_pred - y)
    abs_diff_vals = np.abs(diff_values)
    sorted_indexes = sorted(range(len(abs_diff_vals)), key = lambda x: abs_diff_vals[x])
    sorted_indexes_lead = sorted_indexes[:int(len(abs_diff_vals)*m)]
    return X[sorted_indexes_lead], y[sorted_indexes_lead], dates[sorted_indexes_lead]


class BoxcoxModel(object):
    def __init__(self, **kwargs):
        #print("kwargs=", kwargs)
        self.is_boxcox = kwargs.get("is_boxcox", False)
        self.boxcox_lambda = kwargs.get("boxcox_lambda", 0.0)
        self.Model = kwargs.get("model", GradientBoostingRegressor)
        if "is_boxcox" in kwargs:
            kwargs.pop("is_boxcox")
        if "boxcox_lambda" in kwargs:
            kwargs.pop("boxcox_lambda")
        if "model" in kwargs:
            kwargs.pop("model")
        self.clf = self.Model(**kwargs)
    def fit(self, X, y):
        if self.is_boxcox:
            self.clf.fit(X, stats.boxcox(y, self.boxcox_lambda))
        else:
            self.clf.fit(X, y)
    def predict(self, X):
        if self.is_boxcox:
            return invboxcox(self.clf.predict(X), self.boxcox_lambda)
        else:
            return self.clf.predict(X)

class CombineModes(object):
    def __init__(self, **kwargs):
        self.models = copy.deepcopy(kwargs.get("models", None))
        self.dates_train = copy.deepcopy(kwargs.get("dates_train", None))
        self.weights = copy.deepcopy(kwargs.get("weights", None))
        if self.weights is not None:
            self.weights = np.asarray(self.weights)
            self.weights = self.weights / np.sum(self.weights)
        self.harmonic_mean = kwargs.get("harmonic_mean", True)
        self.subsample = kwargs.get("subsample", 0.8)
        self.combine_method = kwargs.get("combine_method", 0)
        self.sample_weight = kwargs.get("sample_weight", None)
        self.cache_file = kwargs.get("cache_file", "not useful any more")
        self.model_hash_input_fit_key = []
        self.cache_data = {}
        self.is_save_cache_to_disk = True ### should be true otherwise all the model will be re-initialize each time
        if self.models is not None:
            self.model_hash_input_fit_key = [None] * len(self.models)

    def load_cache_data(self, ):
        global global_variables
        if "cache_combine_model" not in global_variables:
            global_variables["cache_combine_model"] = {}
        self.cache_data.update(global_variables["cache_combine_model"])
        #print("self.cache_data=", self.cache_data)

    def save_cache_data(self, ):
        global global_variables
        if "cache_combine_model" not in global_variables:
            global_variables["cache_combine_model"] = {}
        global_variables["cache_combine_model"].update(self.cache_data)
        #print("self.cache_data=", self.cache_data)

    def fit(self, X, y):
        self.X_train = np.copy(X)
        self.y_train = np.copy(y)

    def _fit(self, X, y, model_i):
       mX = np.copy(X)
       my = np.copy(y)
       sub_index = range(len(my))
       shuffle(sub_index)
       if self.subsample < 1.0:
           sub_index = sub_index[:int(len(my)*self.subsample)]
       sub_index = np.asarray(sub_index)
       sub_index.sort()
       mX = mX[sub_index, :]
       my = my[sub_index]
       dates_train = copy.deepcopy(self.dates_train[sub_index])
       if hasattr(self.models[model_i], "dates_train"):
           self.models[model_i].dates_train = copy.deepcopy(dates_train)
       self.models[model_i].fit(mX, my)

    def _fit_predict(self, X, model_i):
        if self.cache_file is not None:
            self.X_train.flags.writeable = False
            self.y_train.flags.writeable = False
            self.dates_train.flags.writeable = False
            self.model_hash_input_fit_key[model_i] = str(hash(repr(self.models[model_i]))) + \
                                                     str(hash(self.X_train.data)) + \
                                                     str(hash(self.y_train.data)) + \
                                                     str(hash(self.dates_train.data))
            self.X_train.flags.writeable = True
            self.y_train.flags.writeable = True
            self.dates_train.flags.writeable = True

            X.flags.writeable = False
            model_hash_predict_key = str(hash(X.data))
            X.flags.writeable = True
            total_key = self.model_hash_input_fit_key[model_i] + model_hash_predict_key
            if total_key in self.cache_data:
                #print("using cache ", total_key)
                return np.asarray(self.cache_data[total_key])
        self._fit(self.X_train, self.y_train, model_i)
        ret_val = self.models[model_i].predict(X)
        if self.cache_file is not None:
            self.cache_data[total_key] = ret_val.tolist()
            self.save_cache_data()
        return ret_val

    def predict(self, X):
        self.load_cache_data()
        ret_val = None
        if self.combine_method == 0:
            ys = []
            for i in range(len(self.models)):
                y_i = self._fit_predict(X, i)
                ys.append(y_i)
            ret_val = compute_harmonic_mean(ys)
        elif self.combine_method == 1:
            ys = []
            for i in range(len(self.models)):
                y_i = self._fit_predict(X, i)
                ys.append(y_i)
            ys = np.asarray(ys)
            ret_val = np.average(ys, axis=0)
        elif self.combine_method == 2:
            if self.weights is not None:
                sum_w = np.sum(self.weights)
                self.weights = self.weights / sum_w
                y = []
                for i in range(len(self.models)):
                    y_i = self._fit_predict(X, i)
                    y_i = np.asarray(y_i, np.float32)
                    y_i *= np.asarray(self.weights[i], np.float32)
                    y.append(y_i.reshape(-1))
                y = np.asarray(y)
                #print(y.shape)
                y = np.sum(y, axis=0)
                ret_val = y
        return ret_val
          

class DaterangeModel(object):
    def is_in_predict_hour_range(self, item_datetime):
        for i in self.predict_hour_range:
            start_time = datetime(item_datetime.year, item_datetime.month, item_datetime.day, i[0][0], i[0][1])
            end_time = datetime(item_datetime.year, item_datetime.month, item_datetime.day, i[1][0], i[1][1])
            if item_datetime >= start_time and item_datetime < end_time:
                return True
        return False

    def __str__(self, ):
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        ret_val = ""
        for attr in members:
            value = getattr(self, attr)
            ret_val += "    (%s:%s)\n" % (attr, repr(value))
        return "DaterangeModel: " + repr(self.clf) + " " + ret_val

    def __repr__(self, ):
        return str(self)

    def __init__(self, **kwargs):
        #print("kwargs=", kwargs)
        self.dates_train = copy.deepcopy(kwargs.get("dates_train", None))
        self.skip_date_ranges = copy.deepcopy(kwargs.get("skip_date_ranges", []))
        self.ft_select = copy.deepcopy(kwargs.get("ft_select", None))
        self.train_days = kwargs.get("train_days", None)
        self.is_rm_outliers = kwargs.get("is_rm_outliers", None)
        self.is_y_log = kwargs.get("is_y_log", False)
        self.rm_outliers_m = kwargs.get("rm_outliers_m", 3.0)
        self.Model = kwargs.get("model", GradientBoostingRegressor)
        self.rm_outliers_key = kwargs.get("rm_outliers_key", [0, ])
        self.is_avg_or_median = kwargs.get("is_avg_or_median", True)
        self.is_boxcox = kwargs.get("is_boxcox", False)
        self.boxcox_lambda = kwargs.get("boxcox_lambda", False)
        self.y_log_e = kwargs.get("y_log_e", np.e)
        self.random_state = kwargs.get("random_state", None)
        self.anova_filter = kwargs.get("anova_filter", 0)
        self.norm_y = kwargs.get("norm_y", False)
        self.is_one_hot_encode = kwargs.get("is_one_hot_encode", False)
        self.is_ft_union = kwargs.get("is_ft_union", None)
        self.ft_th = kwargs.get("ft_th", None)
        self.ft_weights = kwargs.get("ft_weights", None)
        self.is_sample_weight = kwargs.get("is_sample_weight", None)
        self.remove_non_predict_hour_range = kwargs.get("remove_non_predict_hour_range", False)
        self.remove_test_date_data = kwargs.get("remove_test_date_data", False)
        self.is_ignore_skip_date_count = kwargs.get("is_ignore_skip_date_count", False)
        self.rm_n_head_days = kwargs.get("rm_n_head_days", 0)
        self.rm_n_head_days_hours = kwargs.get("rm_n_head_days_hours", [(0, 6), (10, 15), (20, 22)])
        self.predict_hour_range = kwargs.get("predict_hour_range",
                                                [
                                                   [[8, 0], [10, 0]],
                                                   [[17, 0], [19, 0]],
                                                ]
                                            )
        self.remove_outliers_by_classifier = kwargs.get("remove_outliers_by_classifier", None)

        self.ft_norm = kwargs.get("ft_norm", [])
        self.ft_norm_clfs = []
        # ft_norm = [0, 1]
        #self.predict_hour_range = [
        #     [[8, 0], [10, 0]],
        #     [[17, 0], [19, 0]],
        #]

        if self.Model is None:
            self.Model = GradientBoostingRegressor
        if "rm_n_head_days_hours" in kwargs:
            kwargs.pop("rm_n_head_days_hours")
        if "remove_outliers_by_classifier" in kwargs:
            kwargs.pop("remove_outliers_by_classifier")
        if "rm_n_head_days" in kwargs:
            kwargs.pop("rm_n_head_days")
        if "ft_norm" in kwargs:
            kwargs.pop("ft_norm")
        if "is_sample_weight" in kwargs:
            kwargs.pop("is_sample_weight")
        if "ft_select" in kwargs:
            kwargs.pop("ft_select")
        if "is_rm_outliers" in kwargs:
            kwargs.pop("is_rm_outliers")
        if "rm_outliers_m" in kwargs:
            kwargs.pop("rm_outliers_m")
        if "dates_train" in kwargs:
            kwargs.pop("dates_train")
        if "model" in kwargs:
            kwargs.pop("model")
        if "train_days" in kwargs:
            kwargs.pop("train_days")
        if "rm_outliers_key" in kwargs:
            kwargs.pop("rm_outliers_key")
        if "is_y_log" in kwargs:
            kwargs.pop("is_y_log")
        if "y_log_e" in kwargs:
            kwargs.pop("y_log_e")
        if "is_avg_or_median" in kwargs:
            kwargs.pop("is_avg_or_median")
        if "is_boxcox" in kwargs:
            kwargs.pop("is_boxcox")
        if "boxcox_lambda" in kwargs:
            kwargs.pop("boxcox_lambda")
        if "anova_filter" in kwargs:
            kwargs.pop("anova_filter")
        if "norm_y" in kwargs:
            kwargs.pop("norm_y")
        if "is_one_hot_encode" in kwargs:
            kwargs.pop("is_one_hot_encode")
        if "is_ft_union" in kwargs:
            kwargs.pop("is_ft_union")
        if "ft_th" in kwargs:
            kwargs.pop("ft_th")
        if "ft_weights" in kwargs:
            kwargs.pop("ft_weights")
        if "remove_non_predict_hour_range" in kwargs:
            kwargs.pop("remove_non_predict_hour_range")
        if "predict_hour_range" in kwargs:
            kwargs.pop("predict_hour_range")
        if "remove_test_date_data" in kwargs:
            kwargs.pop("remove_test_date_data")
        if "is_ignore_skip_date_count" in kwargs:
            kwargs.pop("is_ignore_skip_date_count")
        if "skip_date_ranges" in kwargs:
            kwargs.pop("skip_date_ranges")
        if "random_state" in kwargs:
            kwargs.pop("random_state")
        #print inspect.getargspec(self.Model.__init__)
        arguments = inspect.getargspec(self.Model.__init__)[0]
        if "random_state" in arguments:
            kwargs["random_state"] = self.random_state
        self.clf = self.Model(**kwargs)
        #if hasattr(self.clf, "random_state"):
        #    self.clf.random_state = random_state
        if self.anova_filter > 0:
            anova_filter_clf = SelectKBest(f_regression, k=self.anova_filter)
            self.clf = make_pipeline(anova_filter_clf, self.clf)
        if self.is_one_hot_encode:
            self.enc = preprocessing.OneHotEncoder()


    def is_need_skip_n_head_days_hours(self, cur_date): 
        #self.rm_n_head_days_hours = kwargs.get("rm_n_head_days_hours", [(0, 8), (10, 17), (19, 22)])
        for hour_range in self.rm_n_head_days_hours:
            if cur_date.hour >= hour_range[0] and cur_date.hour < hour_range[1]:
                return True
        return False

    def is_need_skip(self, cur_date):
        for skip_date_range in self.skip_date_ranges:
            if cur_date >= skip_date_range[0] and cur_date < skip_date_range[1]:
                return True
        return False

    def fit(self, X, y):
        X = np.copy(X)
        y = np.copy(y)
        if self.ft_th is not None:
            for item in self.ft_th:
              ft_pos = item[0]
              th = item[1]
              print("ft_pos=", ft_pos)
              print("X.shape=", X.shape)
              print(X[:5, ft_pos])
              #print(np.sum(X[:, ft_pos]))
              positive_items = X[:, ft_pos] >= th
              negative_items = np.logical_not(positive_items)
              X[positive_items, ft_pos] = 1
              X[negative_items, ft_pos] = 0
              #print(np.sum(X[:, ft_pos]))
        for pos in self.ft_norm:
            min_max_scaler = preprocessing.MinMaxScaler()
            min_max_scaler.fit(X[:, pos].reshape(-1, 1))
            X[:, pos] = min_max_scaler.transform(X[:, pos].reshape(-1, 1)).reshape(-1)
            self.ft_norm_clfs.append(min_max_scaler)
        #tmp_dates_train = copy.deepcopy(self.dates_train)
        #print(X[:5, :])
        if self.ft_select is not None:
            self.ft_select = np.asarray(self.ft_select)
            self.ft_select = self.ft_select.reshape((-1, ))
            X = X[:, self.ft_select]
        if self.is_y_log:
            y = np.log(y) / np.log(self.y_log_e)
        elif self.is_boxcox:
            y = boxcox(y, self.boxcox_lambda)
        if self.norm_y:
            self.norm_y_max_y = np.max(y)
            self.norm_y_min_y = np.min(y)
            y = (y-self.norm_y_min_y)/(self.norm_y_max_y-self.norm_y_min_y)
        if not all(self.dates_train[i] <= self.dates_train[i+1]
                for i in xrange(len(self.dates_train)-1)):
            raise ValueError("train dates are not sorted...")
        tmp_dates_train = copy.deepcopy(self.dates_train)
        if self.is_rm_outliers:
            if self.dates_train is None:
                raise ValueError("self.dates_train is None")
            X, y, tmp_dates_train = remove_outliers3(
                X, y, tmp_dates_train, self.rm_outliers_m,
                key=self.rm_outliers_key,
                is_avg_or_median=self.is_avg_or_median)
        if self.ft_weights is not None:
            self.ft_weights = np.asarray(self.ft_weights)
            X = np.multiply(X, np.tile(self.ft_weights, (X.shape[0], 1)))
        if self.remove_outliers_by_classifier is not None:
            X, y, tmp_dates_train = remove_outliers_by_classifier(X, y, tmp_dates_train, **self.remove_outliers_by_classifier)
        if self.is_one_hot_encode:
            self.enc.fit(X)
            X = self.enc.transform(X).toarray()
            #print("X[:5,:]=", X[:5,:])
        if self.is_ft_union is not None:
            print("X[:5,:]=", X[:5,:])
            print("X.shape=", X.shape)
            self.is_ft_union.fit(X,y)
            X_ft_u =self.is_ft_union.transform(X)
            X = np.hstack([X, X_ft_u])

        i_train_days = 0
        train_min_date = tmp_dates_train[-1]
        while i_train_days < self.train_days and train_min_date >= tmp_dates_train[0]:
            if not self.is_need_skip(train_min_date) or (self.is_ignore_skip_date_count):
                i_train_days += 1
            train_min_date -= timedelta(days=1)

        train_min_date += timedelta(days=1)
        print("i_train_days=", i_train_days)
        print("real diff days=", (tmp_dates_train[-1] - train_min_date).days)
        train_items = tmp_dates_train >= train_min_date
        X_train = X[train_items, :]
        y_train = y[train_items]
        tmp_dates_train_left = tmp_dates_train[train_items]
        if self.remove_non_predict_hour_range:
            hour_range_items = []
            for item_date in tmp_dates_train_left:
                if self.is_in_predict_hour_range(item_date):
                    hour_range_items.append(True)
                else:
                    hour_range_items.append(False)
            hour_range_items = np.asarray(hour_range_items)
            X_train = X_train[hour_range_items]
            y_train = y_train[hour_range_items]
            tmp_dates_train_left = tmp_dates_train_left[hour_range_items]
            #print("tmp_dates_train_left=", tmp_dates_train_left[-60:])

        if self.rm_n_head_days > 0:
            max_train_day = tmp_dates_train_left[-1] - timedelta(days=self.rm_n_head_days)

            n_head_days_items = []
            for item_date in tmp_dates_train_left:
                if ( (not self.is_need_skip_n_head_days_hours(item_date)) and item_date >= max_train_day) \
                        or item_date < max_train_day:
                    n_head_days_items.append(True)
                else:
                    n_head_days_items.append(False)
            n_head_days_items = np.asarray(n_head_days_items)
            X_train = X_train[n_head_days_items]
            y_train = y_train[n_head_days_items]
            tmp_dates_train_left = tmp_dates_train_left[n_head_days_items]

        if len(self.skip_date_ranges) > 0:
            left_date_range_items = []
            for item_date in tmp_dates_train_left:
                if not self.is_need_skip(item_date):
                    left_date_range_items.append(True)
                else:
                    left_date_range_items.append(False)
            left_date_range_items = np.asarray(left_date_range_items)
            X_train = X_train[left_date_range_items]
            y_train = y_train[left_date_range_items]
            tmp_dates_train_left = tmp_dates_train_left[left_date_range_items]

        print("date range min train dates:", tmp_dates_train_left[0])
        print("date range max train dates:", tmp_dates_train_left[-1])
        print("self.clf.name=", self.clf)
        #self.dates_train = self.dates_train[train_items]
        arguments = inspect.getargspec(self.clf.fit)[0]
        if "sample_weight" in arguments and self.is_sample_weight > 0 and self.is_sample_weight is not None and self.is_sample_weight:
            #print("use sample_weight")
            sample_weight = []
            for datei, tdate in enumerate(tmp_dates_train_left):
                if self.dates_train[-1] >= tdate:
                    div_factor = int((self.dates_train[-1] - tdate).days/self.is_sample_weight) + 1
                else:
                    div_factor = 1.0
                #sample_weight.append(1.0/(np.log(div_factor) + 1.0))
                sample_weight.append(1.0/div_factor)
            sample_weight = np.asarray(sample_weight)
            #sample_weight = (np.max(sample_weight) - sample_weight) / (np.max(sample_weight) - np.min(sample_weight))*100.0
            self.clf.fit(X_train, y_train, sample_weight)
        else:
            self.clf.fit(X_train, y_train)
        if hasattr(self.clf, "feature_importances_"):
            print(self.clf.feature_importances_)

    def predict(self, X):
        X = np.copy(X)
        if self.ft_th is not None:
            for item in self.ft_th:
              ft_pos = item[0]
              th = item[1]
              #print(np.sum(X[:, ft_pos]))
              positive_items = X[:, ft_pos] >= th
              negative_items = np.logical_not(positive_items)
              X[positive_items, ft_pos] = 1
              X[negative_items, ft_pos] = 0
              #print(np.sum(X[:, ft_pos]))
        for i, pos in enumerate(self.ft_norm):
            min_max_scaler = self.ft_norm_clfs[i]
            X[:, pos] = min_max_scaler.transform(X[:, pos].reshape(-1, 1)).reshape(-1)
            
        if self.ft_weights is not None:
            self.ft_weights = np.asarray(self.ft_weights)
            X = np.multiply(X, np.tile(self.ft_weights, (X.shape[0], 1)))
        if self.ft_select is not None:
            self.ft_select = np.asarray(self.ft_select)
            X = X[:, self.ft_select]
        if self.is_one_hot_encode:
            X = self.enc.transform(X).toarray()
        if self.is_ft_union is not None:
            X_ft_u = self.is_ft_union.transform(X)
            X = np.hstack([X,X_ft_u])
        pre_y = self.clf.predict(X)
        if self.norm_y:
            pre_y = pre_y * (self.norm_y_max_y - self.norm_y_min_y) + self.norm_y_min_y
        if self.is_y_log:
            pre_y = np.exp(pre_y * np.log(self.y_log_e))
        elif self.is_boxcox:
            pre_y =  invboxcox(pre_y, self.boxcox_lambda)
        else:
            pre_y = pre_y
        return pre_y


class TestModel(object):
    def __init__(self,):
        pass

    def fit(self, X, y):
        self.clf = ExtraTreesRegressor()
        #y = np.log(y)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

