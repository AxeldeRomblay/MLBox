# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from copy import copy as make_copy
from .estimator import Estimator
import warnings


class StackingEstimator():

    """A stacking estimator.

    A stacking estimator is an estimator that uses the predictions of
    several first layer estimators (generated with a cross validation method)
    for a second layer estimator.

    Parameters
    ----------
    task : str, default = None
        The task ("classification" or "regression")

    base_estimators : list, default = []
        List of estimators to fit in the first level using a cross validation.
        If the list is empty, base estimators are : "LightGBM", "RandomForest" and "ExtraTrees"

    level_estimator : object, default = None
        The estimator used in second and last level.
        If None, the level estimator is : "Linear".

    n_folds : int, default = 5
        Number of folds used to generate the meta features for the training set

    copy : bool, default = False
        If true, meta features are added to the original dataset

    random_state : None or int or RandomState. default = 1
        Pseudo-random number generator state used for shuffling. If None, use
        default numpy RNG for shuffling.

    verbose : bool, default = True
        Verbose mode.
    """

    def __init__(self,
                 task=None,
                 base_estimators=[],
                 level_estimator=None,
                 n_folds=5,
                 copy=False,
                 random_state=1,
                 verbose=True):

        # task
        self.task = task

        if ((self.task != "classification") & (self.task != "regression")):
            raise ValueError("Invalid task ! Please choose between 'classification' or 'regression'.")
        else:
            pass

        # base estimators
        self.base_estimators = base_estimators

        if(type(self.base_estimators) != list):
            raise ValueError("base_estimators must be a list")

        if(len(self.base_estimators)==0):
            self.base_estimators = [Estimator(task=self.task, strategy="LightGBM"),
                                    Estimator(task=self.task, strategy="RandomForest"),
                                    Estimator(task=self.task, strategy="ExtraTrees")]

        for i, est in enumerate(self.base_estimators):
            self.base_estimators[i].set_params({"task": self.task})
            self.base_estimators[i] = make_copy(est)

        # level estimator
        self.level_estimator = level_estimator

        if(self.level_estimator==None):
            self.level_estimator = Estimator(task=self.task, strategy="Linear")

        self.level_estimator.set_params({"task": self.task})

        # folds
        self.n_folds = n_folds
        if(type(self.n_folds) != int):
            raise ValueError("n_folds must be an integer")

        # copy
        self.copy = copy
        if(type(self.copy) != bool):
            raise ValueError("copy must be a boolean")

        # random state
        self.random_state = random_state
        if((type(self.random_state) != int) and
           (self.random_state is not None)):
            raise ValueError("random_state must be either None or an integer")

        # verbose
        self.verbose = verbose
        if(type(self.verbose) != bool):
            raise ValueError("verbose must be a boolean")

        self.__fitOK = False
        self.__fittransformOK = False


    def get_params(self, deep=True):

        return {'task' : self.task,
                'level_estimator': self.level_estimator,
                'base_estimators': self.base_estimators,
                'n_folds': self.n_folds,
                'copy': self.copy,
                'random_state': self.random_state,
                'verbose': self.verbose}


    def set_params(self, **params):

        self.__fitOK = False
        self.__fittransformOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for model StackingEstimator. "
                              "Parameter IGNORED. Check the list of available parameters "
                              "with `stacking_estimator.get_params().keys()`")
            else:
                setattr(self, k, v)


    def __cross_val_predict_proba(self, estimator, df, y, cv):

        """Evaluates the target by cross-validation

        Parameters
        ----------
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        df : pandas DataFrame
            The data to fit.

        y : pandas Serie
            The target variable to try to predict in the case of
            supervised learning.

        cv : a STRATIFIED cross-validation generator


        Returns
        -------
        y_pred : array-like of shape = (n_samples, n_classes)
            The predicted class probabilities for X.
        """

        classes = y.value_counts()
        classes_to_drop = classes[classes < 2].index
        indexes_to_drop = y[y.apply(lambda x: x in classes_to_drop)].index

        y_pred = np.zeros((len(y), len(classes) - len(classes_to_drop)))

        for train_index, test_index in cv.split(df, y):

            # defining train and validation sets for each fold
            df_train, df_test = df.iloc[train_index], df.iloc[test_index]
            y_train = y.iloc[train_index]

            try:
                df_train = df_train.drop(indexes_to_drop)
                y_train = y_train.drop(indexes_to_drop)

            except Exception:
                pass

            # learning the model
            estimator.fit(df_train, y_train)

            # predicting the probability
            y_pred[test_index] = estimator.predict_proba(df_test)[:, ]

        return y_pred


    def fit_transform(self, df_train, y_train):

        """Creates meta-features for the training dataset.

        Parameters
        ----------
        df_train : pandas dataframe of shape = (n_samples, n_features)
            The training dataset.

        y_train : pandas series of shape = (n_samples, )
            The target.

        Returns
        -------
        pandas dataframe of shape = (n_samples, n_features*int(copy)+n_metafeatures)
            The transformed training dataset.
        """

        # sanity checks

        if((type(df_train) != pd.SparseDataFrame) and (type(df_train) != pd.DataFrame)):
            raise ValueError("df_train must be a DataFrame")

        if(type(y_train) != pd.core.series.Series):
            raise ValueError("y_train must be a Series")

        # cv

        if(self.task=="classification"):

            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                                 random_state=self.random_state)

            classes = y_train.value_counts()
            classes_to_drop = classes[classes < 2].index
            indexes_to_drop = y_train[y_train.apply(lambda x: x in classes_to_drop)].index

        else:

            cv = KFold(n_splits=self.n_folds, shuffle=True,
                       random_state=self.random_state)

            indexes_to_drop = []

        # fit

        preds = pd.DataFrame([], index=y_train.index)

        if(self.verbose):
            print("")
            print("[=========================================================="
                  "===================] LAYER [==============================="
                  "====================================================]")
            print("")

        for c, est in enumerate(self.base_estimators):

            if(self.verbose):
                print("> fitting estimator n°" + str(c+1) + " : " +
                      str(est.get_params()) + " ...")
                print("")

            # for each base estimator, we create the meta feature on train set

            if (self.task == "classification"):
                y_pred = self.__cross_val_predict_proba(est, df_train, y_train, cv)
                for i in range(0, y_pred.shape[1]-1):
                    preds["est" + str(c+1) + "_class" + str(i)] = y_pred[:, i]
            else:
                y_pred = cross_val_predict(estimator=est, X=df_train, y=y_train, cv=cv)
                preds["est" + str(c + 1)] = y_pred

            # and we refit the base estimator on entire train set

            est.fit(df_train.drop(indexes_to_drop), y_train.drop(indexes_to_drop))

        layer = 1
        columns = ["layer" + str(layer) + "_" + s for s in preds.columns]
        while(len(np.intersect1d(df_train.columns, columns)) > 0):
            layer = layer + 1
            columns = ["layer" + str(layer) + "_" + s for s in preds.columns]

        preds.columns = ["layer" + str(layer) + "_" + s for s in preds.columns]

        self.__fittransformOK = True

        if(self.copy):
            # we keep also the initial features
            return pd.concat([df_train, preds], axis=1)
        else:
            # we keep only the meta features
            return preds


    def transform(self, df_test):

        """Creates meta-features for the test dataset.

        Parameters
        ----------
        df_test : pandas dataframe of shape = (n_samples_test, n_features)
            The test dataset.

        Returns
        -------
        pandas dataframe of shape = (n_samples_test, n_features*int(copy)+n_metafeatures)
            The transformed test dataset.
        """

        # sanity checks
        if((type(df_test) != pd.SparseDataFrame) and
           (type(df_test) != pd.DataFrame)):
            raise ValueError("df_test must be a DataFrame")

        if(self.__fittransformOK):

            preds_test = pd.DataFrame([], index=df_test.index)

            # for each base estimator, we predict the meta feature on test set
            for c, clf in enumerate(self.base_estimators):
                y_pred_test = clf.predict_proba(df_test)

                for i in range(0, y_pred_test.shape[1] - int(self.drop_first)):
                    idx_name = "est" + str(c+1) + "_class" + str(i)
                    preds_test[idx_name] = y_pred_test[:, i]

            layer = 1
            columns = ["layer" + str(layer) + "_" + s
                       for s in preds_test.columns]

            while(len(np.intersect1d(df_test.columns, columns)) > 0):
                layer = layer + 1
                columns = ["layer" + str(layer) + "_" + s
                           for s in preds_test.columns]

            preds_test.columns = ["layer" + str(layer) + "_" + s
                                  for s in preds_test.columns]

            if(self.copy):
                # we keep also the initial features
                return pd.concat([df_test, preds_test], axis=1)
            else:
                # we keep only the meta features
                return preds_test

        else:
            raise ValueError("Call fit_transform before !")


    def fit(self, df_train, y_train):

        """Fits the first level estimators and the second level estimator on X.

        Parameters
        ----------
        df_train : pandas dataframe of shape (n_samples, n_features)
            Input data

        y_train : pandas series of shape = (n_samples, )
            The target

        Returns
        -------
        object
            self.
        """

        df_train = self.fit_transform(df_train, y_train)  # we fit the base estimators

        if(self.verbose):
            print("")
            print("[=========================================================="
                  "===============] PREDICTION LAYER [========================"
                  "====================================================]")
            print("")
            print("> fitting estimator : ")
            print(str(self.level_estimator.get_params()) + " ...")
            print("")

        # we fit the second level estimator
        self.level_estimator.fit(df_train.values, y_train.values)

        self.__fitOK = True

        return self


    def predict_proba(self, df_test):

        """Predicts class probabilities for the test set using the meta-features.

        Parameters
        ----------
        df_test : pandas DataFrame of shape = (n_samples_test, n_features)
            The testing samples

        Returns
        -------
        array of shape = (n_samples_test, n_classes)
            The class probabilities of the testing samples.
        """

        if(self.__fitOK):
            # we predict the meta features on test set
            df_test = self.transform(df_test)

            # we predict the probability of class 1 using the meta features
            return self.level_estimator.predict_proba(df_test)
        else:

            raise ValueError("Call fit before !")


    def predict(self, df_test):

        """Predicts class for the test set using the meta-features.

        Parameters
        ----------
        df_test : pandas DataFrame of shape = (n_samples_test, n_features)
            The testing samples

        Returns
        -------
        array of shape = (n_samples_test,)
            The predicted classes.
        """

        if(self.__fitOK):
            # we predict the meta features on test set
            df_test = self.transform(df_test)

            # we predict the target using the meta features
            return self.level_estimator.predict(df_test)

        else:

            raise ValueError("Call fit before !")
