# coding: utf-8
# Authors: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
#          Alexis BONDU <alexis.bondu@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

class DriftEstimator():

    """Estimates the drift between two datasets
    
        
    Parameters
    ----------
    estimator : classifier, defaut = RandomForestClassifier(n_estimators = 50, n_jobs=-1, max_features=1., min_samples_leaf = 5, max_depth = 5)
        The estimator that estimates the drift between two datasets
        
    n_folds : int, defaut = 2
        Number of folds used to estimate the drift

    stratify : bool, defaut = True
        Whether the cv is stratified (same number of train and test samples within each fold)

    random_state : int, defaut = 1
        Random state for cv
    """

    def __init__(self,
                 estimator=RandomForestClassifier(n_estimators=50,
                                                  n_jobs=-1,
                                                  max_features=1.,
                                                  min_samples_leaf=5,
                                                  max_depth=5),
                 n_folds=2,
                 stratify=True,
                 random_state=1):

        self.estimator = estimator
        self.n_folds = n_folds
        self.stratify = stratify
        self.random_state = random_state
        self.__cv = None
        self.__pred = None
        self.__target = None
        self.__fitOK = False

    def get_params(self):

        return {'estimator': self.estimator,
                'n_folds': self.n_folds,
                'stratify': self.stratify,
                'random_state': self.random_state}

    def set_params(self, **params):

        if('estimator' in params.keys()):
            self.estimator = params['estimator']
        if('n_folds' in params.keys()):
            self.n_folds = params['n_folds']
        if('stratify' in params.keys()):
            self.stratify = params['stratify']
        if('random_state' in params.keys()):
            self.random_state = params['random_state']

    def fit(self, df_train, df_test):

        """
        Computes the drift between the two datasets

        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_train, p)
            The train set

        df_test : pandas dataframe of shape = (n_test, p)
            The test set

        Returns
        -------
        self : object
            Returns self.
        """

        df_train["target"] = 0
        df_test["target"] = 1

        self.__target = pd.concat((df_train.target, df_test.target),
                                  ignore_index=True)

        if self.stratify:
            self.__cv = StratifiedKFold(n_splits=self.n_folds,
                                        shuffle=True,
                                        random_state=self.random_state)
        else:
            self.__cv = KFold(n_splits=self.n_folds,
                              shuffle=True,
                              random_state=self.random_state)

        X_tmp = pd.concat((df_train, df_test),
                          ignore_index=True).drop(['target'], axis=1)

        self.__pred = cross_val_predict(estimator=self.estimator,
                                        X=X_tmp,
                                        y=self.__target,
                                        cv=self.__cv,
                                        method="predict_proba")

        del df_train["target"]
        del df_test["target"]

        self.__fitOK = True

        return self

    def score(self):
        
        """Returns the global drift measure between two datasets.

         0. = No drift. 1. = Maximal Drift

        Returns
        -------
        float
            The drift measure
        """

        S = []

        if self.__fitOK:

            X_zeros = np.zeros(len(self.__target))

            for train_index, test_index in self.__cv.split(X=X_zeros,
                                                           y=self.__target):

                S.append(roc_auc_score(self.__target.iloc[test_index],
                                       self.__pred[test_index]))

            return (max(np.mean(S), 1-np.mean(S))-0.5) * 2

        else:
            raise ValueError('Call the fit function before !')

    def predict(self):

        """Returns the probabilities that the sample belongs to the test dataset

        Returns
        -------
        Array of shape = (n_train+n_test,)
            The probabilities
        """

        if self.__fitOK:

            return self.__pred

        else:
            raise ValueError('Call the fit function before !')
