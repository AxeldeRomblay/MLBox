# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings


class Clf_feature_selector():

    """Selects useful features.

    Several strategies are possible (filter and wrapper methods).
    Works for classification problems only (multiclass or binary).

    Parameters
    ----------
    strategy : str, defaut = "l1"
        The strategy to select features.
        Available strategies = {"variance", "l1", "rf_feature_importance"}

    threshold : float, defaut = 0.3
        The percentage of variable to discard according to the strategy.
        Must be between 0. and 1.
    """

    def __init__(self, strategy='l1', threshold=0.3):

        # 'variance','l1, 'rf_feature_importance'
        self.strategy = strategy
        # a float between 0. and 1. defaut : 0.3 ie we drop 0.3 of features
        self.threshold = threshold
        self.__fitOK = False
        self.__to_discard = []


    def get_params(self, deep=True):

        return {'strategy': self.strategy,
                'threshold': self.threshold}


    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for feature selector"
                              "Clf_feature_selector. Parameter IGNORED. Check"
                              "the list of available parameters with"
                              "`feature_selector.get_params().keys()`")
            else:
                setattr(self, k, v)


    def fit(self, df_train, y_train):

        """Fits Clf_feature_selector

        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with numerical features and no NA

        y_train : pandas series of shape = (n_train, )
            The target for classification task. Must be encoded.

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

        if(self.strategy == 'variance'):
            coef = df_train.std()
            abstract_threshold = np.percentile(coef, 100. * self.threshold)
            self.__to_discard = coef[coef < abstract_threshold].index
            self.__fitOK = True

        elif(self.strategy == 'l1'):
            model = LogisticRegression(C=0.01, penalty='l1', solver="saga", n_jobs=-1,
                                       random_state=0)  # to be tuned
            model.fit(df_train, y_train)
            coef = np.mean(np.abs(model.coef_), axis=0)
            abstract_threshold = np.percentile(coef, 100. * self.threshold)
            self.__to_discard = df_train.columns[coef < abstract_threshold]
            self.__fitOK = True

        elif(self.strategy == 'rf_feature_importance'):
            model = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                           random_state=0)  # to be tuned
            model.fit(df_train, y_train)
            coef = model.feature_importances_
            abstract_threshold = np.percentile(coef, 100. * self.threshold)
            self.__to_discard = df_train.columns[coef < abstract_threshold]
            self.__fitOK = True

        else:
            raise ValueError("Strategy invalid. Please choose between "
                             "'variance', 'l1' or 'rf_feature_importance'")

        return self


    def transform(self, df):

        """Transforms the dataset

        Parameters
        ----------
        df : pandas dataframe of shape = (n, n_features)
            The dataset with numerical features and no NA

        Returns
        -------
        pandas dataframe of shape = (n_train, n_features*(1-threshold))
            The train dataset with relevant features
        """

        if(self.__fitOK):

            # sanity checks
            if((type(df) != pd.SparseDataFrame) and
               (type(df) != pd.DataFrame)):
                raise ValueError("df must be a DataFrame")

            return df.drop(self.__to_discard, axis=1)
        else:
            raise ValueError("call fit or fit_transform function before")


    def fit_transform(self, df_train, y_train):

        """Fits Clf_feature_selector and transforms the dataset
    
        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with numerical features and no NA

        y_train : pandas series of shape = (n_train, ). 
            The target for classification task. Must be encoded.
    
        Returns
        -------
        pandas dataframe of shape = (n_train, n_features*(1-threshold))
            The train dataset with relevant features
        """

        self.fit(df_train, y_train)

        return self.transform(df_train)
