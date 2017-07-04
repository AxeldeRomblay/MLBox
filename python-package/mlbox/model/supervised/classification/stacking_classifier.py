# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from copy import copy as make_copy
from .classifier import *
import time


class StackingClassifier():

    """A Stacking classifier is a classifier that uses the predictions of several first layer estimators (generated with a cross validation method)
    for a second layer estimator.


    Parameters
    ----------

    base_estimators : list
        List of estimators to fit in the first level using a cross validation.

    level_estimator : object, optional (default=LogisticRegression())
        The estimator used in second and last level.

    n_folds : int, optional (default=5)
        Number of folds used to generate the meta features for the training set.

    copy : boolean, optional (default=False)
        If true, meta features are added to the original dataset

    drop_first = boolean, optional (default=True)
        If True, each estimator output n_classes-1 probabilities

    random_state : None, int or RandomState (default=1)
        Pseudo-random number generator state used for shuffling. If None, use default numpy RNG for shuffling.

    verbose : boolean, optional (default=True)
        Verbose mode.

    """



    def __init__(self, base_estimators = [Classifier(strategy="XGBoost"),Classifier(strategy="RandomForest"),Classifier(strategy="ExtraTrees")], level_estimator = LogisticRegression(n_jobs=-1), n_folds = 5, copy = False, drop_first = True, random_state = 1, verbose = True):


        self.base_estimators = base_estimators
        if(type(self.base_estimators)!=list):
            raise ValueError("base_estimators must be a list")
        else:
            for i, est in enumerate(self.base_estimators):
                self.base_estimators[i] = make_copy(est)

        self.level_estimator = level_estimator

        self.n_folds = n_folds
        if(type(self.n_folds)!=int):
            raise ValueError("n_folds must be an integer")

        self.copy = copy
        if(type(self.copy)!=bool):
            raise ValueError("copy must be a boolean")

        self.drop_first = drop_first
        if(type(self.drop_first)!=bool):
            raise ValueError("drop_first must be a boolean")

        self.random_state = random_state
        if((type(self.random_state)!=int)&(self.random_state!=None)):
            raise ValueError("random_state must be either None or an integer")

        self.verbose = verbose
        if(type(self.verbose)!=bool):
            raise ValueError("verbose must be a boolean")

        self.__fitOK = False
        self.__fittransformOK = False


    def get_params(self, deep = True):

        return {'level_estimator': self.level_estimator,
            'base_estimators' : self.base_estimators,
            'n_folds' : self.n_folds,
            'copy' : self.copy,
            'drop_first' : self.drop_first,
            'random_state' : self.random_state,
            'verbose' : self.verbose}

    def set_params(self,**params):

        self.__fitOK = False
        self.__fittransformOK = False

        for k,v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for stacking_classifier StackingClassifier. Parameter IGNORED. Check the list of available parameters with `stacking_classifier.get_params().keys()`")
            else:
                setattr(self,k,v)

    def __cross_val_predict_proba(self, estimator, X, y, cv):


        """Evaluate the target by cross-validation

    	Parameters
    	----------

    	estimator : estimator object implementing 'fit'
            The object to use to fit the data.

        X : array-like of shape at least 2D
            The data to fit.

        y : array-like
            The target variable to try to predict in the case of
            supervised learning.

        cv : a STRATIFIED cross-validation generator


        Returns
        -------
        y_pred : array-like of shape = [n_samples, n_classes]
            The predicted class probabilities for X.

        """


        classes = y.value_counts()
        classes_to_drop = classes[classes<2].index
        indexes_to_drop = y[y.apply(lambda x: x in classes_to_drop)].index

        y_pred = np.zeros((len(y), len(classes)-len(classes_to_drop)))

        for train_index, test_index in cv.split(X,y):

            X_train, X_test = X.iloc[train_index], X.iloc[test_index]   #defining train et validation sets for each fold
            y_train = y.iloc[train_index]

            try:
                X_train = X_train.drop(indexes_to_drop)
                y_train = y_train.drop(indexes_to_drop)

            except:
                pass

            estimator.fit(X_train, y_train)    #learning the model

            y_pred[test_index] = estimator.predict_proba(X_test)[:,]   #predicting the probability

        return y_pred



    def fit_transform(self, X, y):

        """Create meta-features for the training dataset.

        Parameters
        ----------
        X : DataFrame, shape = [n_samples, n_features]
            The training dataset.

        y : pandas series of shape = [n_samples, ]
            The target.

        Returns
        -------
        X_transform : DataFrame, shape = [n_samples, n_features*int(copy)+n_metafeatures]
            Returns the transformed training dataset.

        """

        ### sanity checks
        if((type(X)!=pd.SparseDataFrame)|(type(X)!=pd.DataFrame)):
            raise ValueError("X must be a DataFrame")

        if(type(y)!=pd.core.series.Series):
            raise ValueError("y must be a Series")


        cv = StratifiedKFold(n_splits = self.n_folds,shuffle=True,random_state=self.random_state)     #stratified k fold

        preds = pd.DataFrame([], index=y.index)

        classes = y.value_counts()
        classes_to_drop = classes[classes<2].index
        indexes_to_drop = y[y.apply(lambda x: x in classes_to_drop)].index

        if(self.verbose):
            print("")
            print("[=============================================================================] LAYER [===================================================================================]")
            print("")

        for c, clf in enumerate(self.base_estimators):

            if(self.verbose):
                print("> fitting estimator n°"+ str(c+1) + " : "+ str(clf.get_params())+" ...")
                print("")

            y_pred = self.__cross_val_predict_proba(clf, X, y, cv)        #for each base estimator, we create the meta feature on train set
            for i in range(0, y_pred.shape[1]-int(self.drop_first)):
                preds["est"+str(c+1)+"_class"+str(i)] = y_pred[:,i]

            clf.fit(X.drop(indexes_to_drop), y.drop(indexes_to_drop))      # and we refit the base estimator on entire train set

        layer = 1
        while(len(np.intersect1d(X.columns, ["layer"+str(layer)+"_"+s for s in preds.columns]))>0):
            layer = layer + 1
        preds.columns = ["layer"+str(layer)+"_"+s for s in preds.columns]
            
        self.__fittransformOK = True

        if(self.copy==True):
            return pd.concat([X, preds], axis=1)     #we keep also the initial features
        else:
            return preds     #we keep only the meta features



    def transform(self, X_test):

        """Create meta-features for the test dataset.

        Parameters
        ----------
        X_test : DataFrame, shape = [n_samples_test, n_features]
            The test dataset.

        Returns
        -------
        X_test_transform : DataFrame, shape = [n_samples_test, n_features*int(copy)+n_metafeatures]
            Returns the transformed test dataset.

        """

        ### sanity checks
        if((type(X_test)!=pd.SparseDataFrame)|(type(X_test)!=pd.DataFrame)):
            raise ValueError("X_test must be a DataFrame")


        if(self.__fittransformOK):

            preds_test = pd.DataFrame([], index=X_test.index)

            for c, clf in enumerate(self.base_estimators):
                y_pred_test = clf.predict_proba(X_test)         #for each base estimator, we predict the meta feature on test set

                for i in range(0, y_pred_test.shape[1]-int(self.drop_first)):
                    preds_test["est"+str(c+1)+"_class"+str(i)] = y_pred_test[:,i]

            layer = 1
            while(len(np.intersect1d(X_test.columns, ["layer"+str(layer)+"_"+s for s in preds_test.columns]))>0):
                layer = layer + 1
            preds_test.columns = ["layer"+str(layer)+"_"+s for s in preds_test.columns]
            
            if(self.copy==True):
                return pd.concat([X_test, preds_test], axis=1)    #we keep also the initial features

            else:
                return preds_test      #we keep only the meta features

        else:
            raise ValueError("Call fit_transform before !")


    def fit(self, X, y):

        """Fit the first level estimators and the second level estimator on X.

        Parameters
        ----------
        X : DataFrame, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y : pandas series of shape = [n_samples, ]
            The target

        Returns
        -------
        self : object
            Returns self.

        """


        X = self.fit_transform(X, y)    #we fit the base estimators

        if(self.verbose):
            print("")
            print("[=========================================================================] PREDICTION LAYER [============================================================================]")
            print("")
            print("> fitting estimator : "+str(self.level_estimator.get_params())+" ...")
            print("")

        self.level_estimator.fit(X.values, y.values)    #we fit the second level estimator

        self.__fitOK = True

        return self


    def predict_proba(self, X_test):

        """Predict class probabilities for X_test using the meta-features.


        Parameters
        ----------
        X_test : DataFrame of shape = [n_samples_test, n_features]
            The testing samples

        Returns
        -------
        p : array of shape = [n_samples_test, n_classes]
            The class probabilities of the testing samples.
        """

        if(self.__fitOK):

            X_test = self.transform(X_test)     # we predict the meta features on test set

            return self.level_estimator.predict_proba(X_test)   #we predict the probability of class 1 using the meta features

        else:

            raise ValueError("Call fit before !")


    def predict(self, X_test):

        """Predict class for X_test using the meta-features.


        Parameters
        ----------
        X_test : DataFrame of shape = [n_samples_test, n_features]
            The testing samples

        Returns
        -------
        y : array of shape = [n_samples_test]
            The predicted classes.
        """

        if(self.__fitOK):

            X_test = self.transform(X_test)   # we predict the meta features on test set

            return self.level_estimator.predict(X_test)     #we predict the target using the meta features

        else:

            raise ValueError("Call fit before !")

