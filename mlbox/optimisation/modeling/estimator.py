# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause


import warnings
from copy import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              BaggingClassifier, BaggingRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

global xgb_installed

try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_installed = True
except Exception:
    warnings.warn(
        "Package xgboost is not installed. Model XGBoost will be replaced by"
        "LightGBM")
    xgb_installed = False

from ..encoding.target_encoder import TargetEncoder


class Estimator():

    """Wraps scikit-learn estimators

    Parameters
    ----------
    task : str, default = None
        The task ("classification" or "regression")

    strategy : str, default = "LightGBM"
        The choice for the estimator.
        Available strategies = "LightGBM", "XGBoost" (if installed),
        "RandomForest", "ExtraTrees", "Tree", "Bagging", "AdaBoost" or "Linear"

    **params : default = None
        Parameters of the corresponding estimator.
        Examples : n_estimators, max_depth...
    """

    def __init__(self, **params):

        # set task
        if ("task" in params):
            self.__task = params["task"]
        else:
            self.__task = None

        # set strategy
        if ("strategy" in params):
            self.__strategy = params["strategy"]
        else:
            if (xgb_installed):
                self.__strategy = "XGBoost"
            else:
                self.__strategy = "LightGBM"

        # init for private variables
        self.__estimator_params = {}
        self.__estimator = None

        # set estimator
        self.__set_estimator(self.__task, self.__strategy)

        # set estimator's parameters
        self.set_params(**params)

        self.__col = None
        self.__fitOK = False


    def get_params(self, deep=True):

        params = {}
        params.update(self.__estimator_params)
        params["strategy"] = self.__strategy
        params["task"] = self.__task

        return params


    def set_params(self, **params):

        self.__fitOK = False

        # set task
        if 'task' in params.keys():
            self.__task = params['task']

        # set strategy then estimator
        if 'strategy' in params.keys():
            self.__strategy = params['strategy']
            self.__set_estimator(self.__task, self.__strategy)

            # try to set old parameters for the new strategy
            for k, v in self.__estimator_params.items():
                try:
                    setattr(self.__estimator, k, v)
                except:
                    warnings.warn("Invalid parameter for estimator " +
                                    str(self.__strategy) +
                                    ". Parameter IGNORED. Check the list of "
                                    "available parameters with "
                                    "`estimator.get_params().keys()`")

        # set estimator's parameters
        for k, v in params.items():
            if((k == "strategy")|(k == "task")):
                pass
            else:
                if self.__estimator==None:
                    warnings.warn("Estimator is not set !")
                else:
                    if k not in self.__estimator.get_params().keys():
                        warnings.warn("Invalid parameter for estimator " +
                                      str(self.__strategy) +
                                      ". Parameter IGNORED. Check the list of "
                                      "available parameters with "
                                      "`estimator.get_params().keys()`")
                    else:
                        setattr(self.__estimator, k, v)
                #and save it
                self.__estimator_params[k] = v


    def __set_estimator(self, task, strategy):

        if(task=="classification"):

            if(strategy == 'RandomForest'):
                self.__estimator = RandomForestClassifier(
                    n_estimators=400, max_depth=10, max_features='sqrt',
                    bootstrap=True, n_jobs=-1, random_state=0)

            elif(strategy == 'XGBoost'):
                if (xgb_installed):
                    self.__estimator = XGBClassifier(n_estimators=500, max_depth=6,
                                                      learning_rate=0.05,
                                                      colsample_bytree=0.8,
                                                      colsample_bylevel=1.,
                                                      subsample=0.9,
                                                      nthread=-1, seed=0)
                else:
                    warnings.warn(
                        "Package xgboost is not installed. Model XGBoost will be"
                        "replaced by LightGBM")
                    self.__strategy = "LightGBM"
                    self.__estimator = LGBMClassifier(
                        n_estimators=500, learning_rate=0.05,
                        colsample_bytree=0.8, subsample=0.9, nthread=-1, seed=0)

            elif(strategy == "LightGBM"):
                self.__estimator = LGBMClassifier(
                    n_estimators=500, learning_rate=0.05,
                    colsample_bytree=0.8, subsample=0.9, nthread=-1, seed=0)

            elif(strategy == 'ExtraTrees'):
                self.__estimator = ExtraTreesClassifier(
                    n_estimators=400, max_depth=10, max_features='sqrt',
                    bootstrap=True, n_jobs=-1, random_state=0)

            elif(strategy == 'Tree'):
                self.__estimator = DecisionTreeClassifier(
                    criterion='gini', splitter='best', max_depth=None,
                    min_samples_split=2, min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0, max_features=None,
                    random_state=0, max_leaf_nodes=None, class_weight=None,
                    presort=False)

            elif(strategy == "Bagging"):
                self.__estimator = BaggingClassifier(
                    base_estimator=None, n_estimators=500, max_samples=.9,
                    max_features=.85, bootstrap=False, bootstrap_features=False,
                    n_jobs=-1, random_state=0)

            elif(strategy == "AdaBoost"):
                self.__estimator = AdaBoostClassifier(
                    base_estimator=None, n_estimators=400, learning_rate=.05,
                    algorithm='SAMME.R', random_state=0)

            elif(strategy == "Linear"):
                self.__estimator = LogisticRegression(
                    penalty='l2', dual=False, tol=0.0001, C=1.0,
                    fit_intercept=True, intercept_scaling=1, class_weight=None,
                    random_state=0, solver='liblinear', max_iter=100,
                    multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1)

            else:
                raise ValueError(
                    "Strategy invalid. Please choose between 'LightGBM', "
                    "'XGBoost' (if installed), 'RandomForest', 'ExtraTrees', "
                    "'Tree', 'Bagging', 'AdaBoost' or 'Linear'")

        elif (task == "regression"):

            if (strategy == 'RandomForest'):
                self.__estimator = RandomForestRegressor(
                    n_estimators=400, max_depth=10, max_features='sqrt',
                    bootstrap=True, n_jobs=-1, random_state=0)

            elif (strategy == 'XGBoost'):
                if (xgb_installed):
                    self.__estimator = XGBRegressor(
                        n_estimators=500, max_depth=6, learning_rate=0.05,
                        colsample_bytree=0.8, colsample_bylevel=1., subsample=0.9,
                        nthread=-1, seed=0)
                else:
                    warnings.warn(
                        "Package xgboost is not installed. Model XGBoost will be"
                        "replaced by LightGBM")
                    self.__strategy = "LightGBM"
                    self.__estimator = LGBMRegressor(
                        n_estimators=500, learning_rate=0.05,
                        colsample_bytree=0.8, subsample=0.9, nthread=-1, seed=0)

            elif (strategy == "LightGBM"):
                self.__estimator = LGBMRegressor(
                    n_estimators=500, learning_rate=0.05,
                    colsample_bytree=0.8, subsample=0.9, nthread=-1, seed=0)

            elif (strategy == 'ExtraTrees'):
                self.__estimator = ExtraTreesRegressor(
                    n_estimators=400, max_depth=10, max_features='sqrt',
                    bootstrap=True, n_jobs=-1, random_state=0)

            elif (strategy == 'Tree'):
                self.__estimator = DecisionTreeRegressor(
                    criterion='mse', splitter='best', max_depth=None,
                    min_samples_split=2, min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0, max_features=None,
                    random_state=0, max_leaf_nodes=None, presort=False)

            elif (strategy == "Bagging"):
                self.__estimator = BaggingRegressor(
                    base_estimator=None, n_estimators=500, max_samples=.9,
                    max_features=.85, bootstrap=False, bootstrap_features=False,
                    n_jobs=-1, random_state=0)

            elif (strategy == "AdaBoost"):
                self.__estimator = AdaBoostRegressor(
                    base_estimator=None, n_estimators=400, learning_rate=.05,
                    random_state=0)

            elif (strategy == "Linear"):
                self.__estimator = Ridge(
                    alpha=1.0, fit_intercept=True, normalize=False, copy_X=True,
                    max_iter=None, tol=0.001, solver='auto', random_state=0)

            else:
                raise ValueError(
                    "Strategy invalid. Please choose between 'LightGBM', "
                    "'XGBoost' (if installed), 'RandomForest', 'ExtraTrees', "
                    "'Tree', 'Bagging', 'AdaBoost' or 'Linear'")

        else:
            warnings.warn("Invalid task ! Please choose between 'classification' or 'regression'. "
                          "Otherwise task will be detected automatically.")

    def get_estimator(self):

        return copy(self.__estimator)

    def fit(self, df_train, y_train):

        """Fits Estimator.

        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with numerical features.

        y_train : pandas series of shape = (n_train,)
            The numerical encoded target for classification tasks.

        Returns
        -------
        object
            self
        """

        # sanity checks
        if((type(df_train) != pd.SparseDataFrame) and
           (type(df_train) != pd.DataFrame)):
            raise ValueError("df_train must be a DataFrame")

        if (type(y_train) != pd.core.series.Series):
            raise ValueError("y_train must be a Series")

        # task detection

        if ((self.__task == "classification") | (self.__task == "regression")):
            detectedtask = self.__task

        else:
            te = TargetEncoder()
            te.detect_task(y_train)
            detectedtask = te.get_task()

        if self.__estimator==None:

            self.__set_estimator(detectedtask, self.__strategy)
            self.set_params(self.__estimator_params)

        # fit
        self.__estimator.fit(df_train.values, y_train)
        self.__col = df_train.columns
        self.__fitOK = True

        return self


    def feature_importances(self):

        """Computes feature importances.

        Estimator must be fitted before.

        Returns
        -------
        dict
            Dictionnary containing a measure of feature importance (value) for
            each feature (key).
        """

        if self.__fitOK:

            if (self.get_params()["strategy"] in ["Linear"]):

                importance = {}
                f = np.mean(np.abs(self.get_estimator().coef_), axis=0)

                for i, col in enumerate(self.__col):
                    importance[col] = f[i]

            elif (self.get_params()["strategy"] in ["LightGBM", "XGBoost",
                                                    "RandomForest",
                                                    "ExtraTrees", "Tree"]):

                importance = {}
                f = self.get_estimator().feature_importances_

                for i, col in enumerate(self.__col):
                    importance[col] = f[i]


            elif(self.get_params()["strategy"] in ["AdaBoost"]):

                importance = {}
                norm = self.get_estimator().estimator_weights_.sum()

                try:
                    # XGB, RF, ET, Tree and AdaBoost
                    f = sum(weight * est.feature_importances_
                        for weight, est in zip(self.get_estimator().estimator_weights_,
                                               self.get_estimator().estimators_)) / norm  # noqa

                except:  # noqa
                    # Linear
                    f = sum(weight * np.mean(np.abs(est.coef_), axis=0)
                        for weight, est in zip(self.get_estimator().estimator_weights_,
                                               self.get_estimator().estimators_)) / norm  # noqa

                for i, col in enumerate(self.__col):
                    importance[col] = f[i]

            elif (self.get_params()["strategy"] in ["Bagging"]):

                importance = {}
                importance_bag = []

                for i, b in enumerate(self.get_estimator().estimators_):

                    d = {}

                    try:
                        # XGB, RF, ET, Tree and AdaBoost
                        f = b.feature_importances_
                    except:  # noqa
                        # Linear
                        f = np.mean(np.abs(b.coef_), axis=0)

                    for j, c in enumerate(self.get_estimator().estimators_features_[i]):  # noqa
                        d[self.__col[c]] = f[j]

                    importance_bag.append(d.copy())

                for i, col in enumerate(self.__col):
                    importance[col] = np.mean(
                        filter(lambda x: x != 0, [k[col] if col in k else 0 for k in importance_bag]))  # noqa

            else:

                importance = {}

            return importance

        else:

            raise ValueError("You must call the fit function before !")


    def predict(self, df):

        """Predicts the target.

        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.

        Returns
        -------
        array of shape = (n, )
            The prediction.
        """

        try:
            if not callable(getattr(self.__estimator, "predict")):
                raise ValueError("predict attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__estimator.predict(df.values)

        else:
            raise ValueError("You must call the fit function before !")

    def predict_log_proba(self, df):

        """Predicts class log-probabilities for df (for classification task only)

        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.

        Returns
        -------
        y : array of shape = (n, n_classes)
            The log-probabilities for each class
        """

        try:
            if not callable(getattr(self.__estimator, "predict_log_proba")):
                raise ValueError("predict_log_proba attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__estimator.predict_log_proba(df.values)
        else:
            raise ValueError("You must call the fit function before !")


    def predict_proba(self, df):

        """Predicts class probabilities for df (for classification task only)

        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.

        Returns
        -------
        array of shape = (n, n_classes)
            The probabilities for each class
        """

        try:
            if not callable(getattr(self.__estimator, "predict_proba")):
                raise ValueError("predict_proba attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__estimator.predict_proba(df.values)
        else:
            raise ValueError("You must call the fit function before !")


    def transform(self, df):

        """Transforms df.

        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.

        Returns
        -------
        pandas dataframe of shape = (n, n_selected_features)
            The transformed dataset with its most important features.
        """

        try:
            if not callable(getattr(self.__estimator, "transform")):
                raise ValueError("transform attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__estimator.transform(df.values)
        else:
            raise ValueError("You must call the fit function before !")


    def score(self, df, y, sample_weight=None):

        """Returns the mean accuracy for classification or the coefficient of determination R^2 ofr regression

        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.

        y : pandas series of shape = (n,)
            The numerical encoded target for classification tasks.

        Returns
        -------
        float
            The score of self.predict(df) wrt. y.
        """

        try:
            if not callable(getattr(self.__estimator, "score")):
                raise ValueError("score attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            if(type(y) != pd.core.series.Series):
                raise ValueError("y must be a Series")

            return self.__estimator.score(df.values, y, sample_weight)
        else:
            raise ValueError("You must call the fit function before !")



