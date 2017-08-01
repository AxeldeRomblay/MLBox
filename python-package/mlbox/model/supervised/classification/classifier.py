# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import warnings
from copy import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, RandomForestClassifier)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

global lgbm_installed

try:
    from lightgbm import LGBMClassifier
    lgbm_installed = True
except Exception:
    warnings.warn(
        "Package lightgbm is not installed. Model LightGBM will be replaced by"
        "XGBoost")
    lgbm_installed = False


class Classifier():

    """Wraps scikitlearn classifiers

    Parameters
    ----------
    strategy : str, default = "LightGBM" if installed else "XGBoost"
        The choice for the classifier.
        Available strategies = "LightGBM" (if installed), "XGBoost",
        "RandomForest", "ExtraTrees", "Tree", "Bagging", "AdaBoost" or "Linear"

    **params : default = None
        Parameters of the corresponding classifier.
        Examples : n_estimators, max_depth...
    """

    def __init__(self, **params):

        if ("strategy" in params):
            self.__strategy = params["strategy"]
        else:
            if (lgbm_installed):
                self.__strategy = "LightGBM"
            else:
                self.__strategy = "XGBoost"

        self.__classif_params = {}

        self.__classifier = None
        self.__set_classifier(self.__strategy)
        self.__col = None

        self.set_params(**params)
        self.__fitOK = False


    def get_params(self, deep=True):

        params = {}
        params["strategy"] = self.__strategy
        params.update(self.__classif_params)

        return params


    def set_params(self, **params):

        self.__fitOK = False

        if 'strategy' in params.keys():
            self.__set_classifier(params['strategy'])

            for k, v in self.__classif_params.items():
                if k not in self.get_params().keys():
                    warnings.warn("Invalid parameter for classifier " +
                                  str(self.__strategy) +
                                  ". Parameter IGNORED. Check the list of "
                                  "available parameters with "
                                  "`classifier.get_params().keys()`")
                else:
                    setattr(self.__classifier, k, v)

        for k, v in params.items():
            if(k == "strategy"):
                pass
            else:
                if k not in self.__classifier.get_params().keys():
                    warnings.warn("Invalid parameter for classifier " +
                                  str(self.__strategy) +
                                  ". Parameter IGNORED. Check the list of "
                                  "available parameters with "
                                  "`classifier.get_params().keys()`")
                else:
                    setattr(self.__classifier, k, v)
                    self.__classif_params[k] = v


    def __set_classifier(self, strategy):

        self.__strategy = strategy

        if(strategy == 'RandomForest'):
            self.__classifier = RandomForestClassifier(
                n_estimators=400, max_depth=10, max_features='sqrt',
                bootstrap=True, n_jobs=-1, random_state=0)

        elif(strategy == 'XGBoost'):
            self.__classifier = XGBClassifier(n_estimators=500, max_depth=6,
                                              learning_rate=0.05,
                                              colsample_bytree=0.8,
                                              colsample_bylevel=1.,
                                              subsample=0.9,
                                              nthread=-1, seed=0)

        elif(strategy == "LightGBM"):
            if(lgbm_installed):
                self.__classifier = LGBMClassifier(
                    n_estimators=500, learning_rate=0.05,
                    colsample_bytree=0.8, subsample=0.9, nthread=-1, seed=0)
            else:
                warnings.warn(
                    "Package lightgbm is not installed. Model LightGBM will be"
                    "replaced by XGBoost")
                self.__strategy = "XGBoost"
                self.__classifier = XGBClassifier(n_estimators=500,
                                                  max_depth=6,
                                                  learning_rate=0.05,
                                                  colsample_bytree=0.8,
                                                  colsample_bylevel=1.,
                                                  subsample=0.9, nthread=-1,
                                                  seed=0)

        elif(strategy == 'ExtraTrees'):
            self.__classifier = ExtraTreesClassifier(
                n_estimators=400, max_depth=10, max_features='sqrt',
                bootstrap=True, n_jobs=-1, random_state=0)

        elif(strategy == 'Tree'):
            self.__classifier = DecisionTreeClassifier(
                criterion='gini', splitter='best', max_depth=None,
                min_samples_split=2, min_samples_leaf=1,
                min_weight_fraction_leaf=0.0, max_features=None,
                random_state=0, max_leaf_nodes=None, class_weight=None,
                presort=False)

        elif(strategy == "Bagging"):
            self.__classifier = BaggingClassifier(
                base_estimator=None, n_estimators=500, max_samples=.9,
                max_features=.85, bootstrap=False, bootstrap_features=False,
                n_jobs=-1, random_state=0)

        elif(strategy == "AdaBoost"):
            self.__classifier = AdaBoostClassifier(
                base_estimator=None, n_estimators=400, learning_rate=.05,
                algorithm='SAMME.R', random_state=0)

        elif(strategy == "Linear"):
            self.__classifier = LogisticRegression(
                penalty='l2', dual=False, tol=0.0001, C=1.0,
                fit_intercept=True, intercept_scaling=1, class_weight=None,
                random_state=0, solver='liblinear', max_iter=100,
                multi_class='ovr', verbose=0, warm_start=False, n_jobs=-1)

        else:
            raise ValueError(
                "Strategy invalid. Please choose between 'LightGBM' "
                "(if installed), 'XGBoost', 'RandomForest', 'ExtraTrees', "
                "'Tree', 'Bagging', 'AdaBoost' or 'Linear'")


    def fit(self, df_train, y_train):

        """Fits Classifier.

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

        self.__classifier.fit(df_train.values, y_train)
        self.__col = df_train.columns
        self.__fitOK = True

        return self


    def feature_importances(self):

        """Computes feature importances.

        Classifier must be fitted before.

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
                        for weight, est in zip(self.get_estimator().estimator_weights_, self.get_estimator().estimators_)) / norm  # noqa

                except:  # noqa
                    # Linear
                    f = sum(weight * np.mean(np.abs(est.coef_), axis=0)
                        for weight, est in zip(self.get_estimator().estimator_weights_, self.get_estimator().estimators_)) / norm  # noqa

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
            The encoded classes to be predicted.
        """

        try:
            if not callable(getattr(self.__classifier, "predict")):
                raise ValueError("predict attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__classifier.predict(df.values)

        else:
            raise ValueError("You must call the fit function before !")

    def predict_log_proba(self, df):

        """Predicts class log-probabilities for df.

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
            if not callable(getattr(self.__classifier, "predict_log_proba")):
                raise ValueError("predict_log_proba attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__classifier.predict_log_proba(df.values)
        else:
            raise ValueError("You must call the fit function before !")


    def predict_proba(self, df):

        """Predicts class probabilities for df.

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
            if not callable(getattr(self.__classifier, "predict_proba")):
                raise ValueError("predict_proba attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__classifier.predict_proba(df.values)
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
            if not callable(getattr(self.__classifier, "transform")):
                raise ValueError("transform attribute is not callable")
        except Exception as e:
            raise e

        if self.__fitOK:

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return self.__classifier.transform(df.values)
        else:
            raise ValueError("You must call the fit function before !")


    def score(self, df, y, sample_weight=None):

        """Returns the mean accuracy.

        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features.

        y : pandas series of shape = (n,)
            The numerical encoded target for classification tasks.

        Returns
        -------
        float
            Mean accuracy of self.predict(df) wrt. y.
        """

        try:
            if not callable(getattr(self.__classifier, "score")):
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

            return self.__classifier.score(df.values, y, sample_weight)
        else:
            raise ValueError("You must call the fit function before !")


    def get_estimator(self):

        return copy(self.__classifier)
