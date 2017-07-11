# coding: utf-8
# Authors: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
#          Alexis BONDU <alexis.bondu@credit-agricole-sa.fr>
# License: BSD 3 clause

import sys
import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def sync_fit(df_train, df_test, estimator, n_folds, stratify, random_state):


    """
    Computes the univariate drifts between df_train and df_test datasets. Multi-threaded version.


    Parameters
    ----------

    df_train : pandas dataframe of shape = (n_train, p)

    df_test : pandas dataframe of shape = (n_test, p)

    estimator : classifier, defaut = DecisionTreeClassifier(max_depth=6)
        The estimator that estimates the drift between two datasets

    n_folds : int, defaut = 2
        Number of folds used to estimate the drift

    stratify : bool, defaut = True
        Whether the cv is stratified (same number of train and test samples within each fold)

    random_state : int, defaut = 1
        Random state for cv

    Returns
    -------
    drift measure (float)

    """

    from .drift_estimator import DriftEstimator

    de = DriftEstimator(estimator, n_folds, stratify, random_state)   #on va recalculer les index de cv sur chaque thread...
    de.fit(df_train,df_test)

    return de.score()


class DriftThreshold():

    """
    Estimates the univariate drift between two datasets and select features with low drifts


    Parameters
    ----------

    threshold : float between 0.5 and 1, defaut = 0.6
        The drift threshold (univariate drift below are kept)

    subsample : float between 0. and 1. , defaut = 1.
        Subsampling parameter for the datasets.

    estimator : classifier, defaut = DecisionTreeClassifier(max_depth=6)
        The estimator that estimates the drift between two datasets

    n_folds : int, defaut = 2
        Number of folds used to estimate the drift

    stratify : bool, defaut = True
        Whether the cv is stratified (same number of train and test samples within each fold)

    random_state : int, defaut = 1
        Seed for for cv and subsampling

    n_jobs : int, defaut = -1
        Number of cores used for processing (-1 -> all cores, else -> 1 core)

    """

    def __init__(self, threshold = 0.6, subsample = 1., estimator = DecisionTreeClassifier(max_depth=6), n_folds = 2, stratify = True, random_state = 1, n_jobs=-1):

        self.threshold = threshold
        self.subsample = subsample
        self.estimator = estimator
        self.n_folds = n_folds
        self.stratify = stratify
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.__Ddrifts = dict()
        self.__fitOK = False

        if(self.n_jobs==-1):
            import ipyparallel as ipp
            self.__client = ipp.Client(profile='home')
            self.__dview = self.__client.direct_view()


    def shutdown_engines(self):

        self.__client.shutdown(hub=True)
        os.system('ipcluster stop --profile=default')


    def get_params(self):

        return {'threshold' : self.threshold,
                'subsample' : self.subsample,
                'estimator' : self.estimator,
                'n_folds' : self.n_folds,
                'stratify' : self.stratify,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs}


    def set_params(self,**params):

        if('threshold' in params.keys()):
            self.threshold = params['threshold']
        if('subsample' in params.keys()):
            self.subsample = params['subsample']
        if('estimator' in params.keys()):
            self.estimator = params['estimator']
        if('n_folds' in params.keys()):
            self.n_folds = params['n_folds']
        if('stratify' in params.keys()):
            self.stratify = params['stratify']
        if('random_state' in params.keys()):
            self.random_state = params['random_state']
        if('n_jobs' in params.keys()):
            self.n_jobs = params['n_jobs']


    def fit(self, df_train, df_test):

        '''
        Computes the univariate drifts between df_train and df_test datasets.


        Parameters
        ----------

        df_train : pandas dataframe of shape = (n_train, p)

        df_test : pandas dataframe of shape = (n_test, p)


        Returns
        -------
        None

        '''


        self.__Ddrifts = dict()

        if((self.subsample != 1.)|(self.random_state != 1)):
            np.random.seed(self.random_state)
            df_train = df_train.iloc[np.random.permutation(np.arange(self.subsample*len(df_train)))]
            df_test = df_test.iloc[np.random.permutation(np.arange(self.subsample*len(df_test)))]

        if(self.n_jobs==-1):

            with self.__dview.sync_imports(quiet=True):

                Ldrifts = self.__dview.map_sync(sync_fit, [df_train[[col]] for col in df_train.columns],[df_test[[col]] for col in df_train.columns], [self.estimator for col in df_train.columns] , [self.n_folds for col in df_train.columns], [self.stratify for col in df_train.columns], [self.random_state for col in df_train.columns])

            for i,col in enumerate(df_train.columns):

                self.__Ddrifts[col] = Ldrifts[i]

            del Ldrifts

        else:
            print("You are using 1 job...")
            
            de = DriftEstimator(estimator = DecisionTreeClassifier(max_depth=6), n_folds = self.n_folds, stratify = self.stratify, random_state = self.random_state)
            
            for col in df_train.columns:

                de.fit(df_train[[col]],df_test[[col]]) 
                self.__Ddrifts[col] = de.score()
                
        self.__fitOK = True
            
        
    def transform(self, df):
                
        """
        Select the features with low drift 
        
        Parameters
        ----------
        
        df : pandas dataframe 
        

        Returns
        -------
        a sub-dataframe  
        
        """
        
        if self.__fitOK:
            
            selected_col = []

            for i, col in enumerate(df.columns):

                if(self.__Ddrifts[col]<self.threshold):
                    selected_col.append(col)

            return df[selected_col]
        
        else:
            raise ValueError('Call the fit function before !') 
    
    
    
    def get_support(self, complement = False):
              
        '''
        Returns the variables kept or dropped.
        '''   
        
        if self.__fitOK:
        
            keepList = []
            dropList = []

            for col in self.__Ddrifts:

                if(self.__Ddrifts[col] < self.threshold):
                    keepList.append(col)
                else:
                    dropList.append(col)

            if(complement):
                return dropList
            else:
                return keepList
        else:
            raise ValueError('Call the fit function before !') 
           
        
        
    def drifts(self):
                        
        '''
        Returns the univariate drifts for all variables.
        '''
        
        if self.__fitOK:
        
            return self.__Ddrifts
        else:
            raise ValueError('Call the fit function before !') 
