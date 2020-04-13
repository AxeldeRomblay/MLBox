# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings

from ..encoding.target_encoder import TargetEncoder

class FeatureSelector():

    """Selects useful features.

    Several strategies are possible (filter and wrapper methods).

    Parameters
    ----------
    task : str, default = "auto"
        The task ("classification" or "regression")

    strategy : str, default = "l1"
        The strategy to select features.
        Available strategies = {"variance", "l1", "rf"}

    threshold : float, default = 0.3
        The percentage of features to discard according to the strategy.
        Must be between 0. and 1.
    """

    def __init__(self, task="auto", strategy='l1', threshold=0.3):

        self.task = task
        self.strategy = strategy
        self.threshold = threshold
        self.__fitOK = False
        self.__to_discard = []


    def get_params(self, deep=True):

        return {'task' : self.task,
                'strategy': self.strategy,
                'threshold': self.threshold}


    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter for selector Feature Selector. "
                              "Parameter IGNORED. Check the list of available "
                              "parameters with `feature_selector.get_params().keys()`")
            else:
                setattr(self, k, v)


    def fit(self, df_train, y_train):

        """Fits Feature_selector

        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with numerical features and no NA

        y_train : pandas series of shape = (n_train, )
            The target

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

        if((self.task=="classification") | (self.task=="regression")):
            detectedtask = self.task

        else:
            if (self.task != "auto"):
                self.task = "auto"
                warnings.warn("Invalid task ! 'auto' is set instead.")
            else:
                pass
            te = Target_encoder()
            te.detect_task(y_train)
            detectedtask = te.get_task()

        # fit

        if(self.strategy == 'variance'):
            coef = df_train.std()
            abstract_threshold = np.percentile(coef, 100. * self.threshold)
            self.__to_discard = coef[coef < abstract_threshold].index
            self.__fitOK = True

        elif(self.strategy == 'l1'):

            if (detectedtask == "classification"):
                model = LogisticRegression(C=0.01, penalty='l1', n_jobs=-1,
                                            random_state=0)  # to be tuned
            else:
                model = Lasso(alpha=100.0, random_state=0)

            model.fit(df_train, y_train)
            coef = np.mean(np.abs(model.coef_), axis=0)
            abstract_threshold = np.percentile(coef, 100. * self.threshold)
            self.__to_discard = df_train.columns[coef < abstract_threshold]
            self.__fitOK = True

        elif(self.strategy == 'rf'):

            if (detectedtask == "classification"):
                model = RandomForestClassifier(n_estimators=50,
                                                n_jobs=-1,
                                                random_state=0)  # to be tuned
            else:
                model = RandomForestRegressor(n_estimators=50,
                                                n_jobs=-1,
                                                random_state=0)
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

        """Fits Feature_selector and transforms the dataset
    
        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, n_features)
            The train dataset with numerical features and no NA

        y_train : pandas series of shape = (n_train, ). 
            The target
    
        Returns
        -------
        pandas dataframe of shape = (n_train, n_features*(1-threshold))
            The train dataset with relevant features
        """

        self.fit(df_train, y_train)

        return self.transform(df_train)
