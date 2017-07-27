
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings

from sklearn.preprocessing import Imputer


class NA_encoder():

    """
    Encodes missing values for both numerical and categorical features.
    Several strategies are possible in each case.


    Parameters
    ----------

    numerical_strategy : string or float or int, default = "mean"
        The strategy to encode NA for numerical features.
        Available strategies = "mean", "median",
                               "most_frequent" or a float/int value

    categorical_strategy : string, default = '<NULL>'
        The strategy to encode NA for categorical features.
        Available strategies = a string or np.NaN
    """

    def __init__(self,
                 numerical_strategy='mean',
                 categorical_strategy='<NULL>'):

        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.__Lcat = []
        self.__Lnum = []
        self.__imp = None
        self.__fitOK = False

    def get_params(self, deep=True):

        return {'numerical_strategy': self.numerical_strategy,
                'categorical_strategy': self.categorical_strategy}

    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for encoder NA_encoder. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`encoder.get_params().keys()`")
            else:
                setattr(self, k, v)

    def fit(self, df_train, y_train=None):

        '''

        Fits NA Encoder.

        Parameters
        ----------

        df_train : pandas dataframe of shape = (n_train, n_features)
        The train dataset with numerical and categorical features.

        y_train : [OPTIONAL]. pandas series of shape = (n_train, ).
            Default = None
            The target for classification or regression tasks.


        Returns
        -------
        None

       '''

        self.__Lcat = df_train.dtypes[df_train.dtypes == 'object'].index
        self.__Lnum = df_train.dtypes[df_train.dtypes != 'object'].index

        if (self.numerical_strategy in ['mean', 'median', "most_frequent"]):

            self.__imp = Imputer(strategy=self.numerical_strategy)

            if (len(self.__Lnum) != 0):
                self.__imp.fit(df_train[self.__Lnum])
            else:
                pass

            self.__fitOK = True

        elif ((type(self.numerical_strategy) == int) | (type(self.numerical_strategy) == float)):

            self.__fitOK = True

        else:

            raise ValueError("Numerical strategy for NA encoding is not valid")

        return self

    def fit_transform(self, df_train, y_train=None):

        '''

        Fits NA Encoder and transforms the dataset.

        Parameters
        ----------

        df_train : pandas.Dataframe of shape = (n_train, n_features)
            The train dataset with numerical and categorical features.

        y_train : [OPTIONAL]. pandas.Series of shape = (n_train, ).
            Default = None.
            The target for classification or regression tasks.


        Returns
        -------

        df_train : pandas.Dataframe of shape = (n_train, n_features)
            The train dataset with no missing values.

        '''

        self.fit(df_train, y_train)

        return self.transform(df_train)

    def transform(self, df):

        '''

        Transforms the dataset

        Parameters
        ----------

        df : pandas.Dataframe of shape = (n, n_features)
            The dataset with numerical and categorical features.


        Returns
        -------

        df : pandas.Dataframe of shape = (n, n_features)
            The dataset with no missing values.

        '''

        if(self.__fitOK):

            if(len(self.__Lnum) == 0):

                return df[self.__Lcat].fillna(self.categorical_strategy)

            else:

                if (self.numerical_strategy in ['mean',
                                                'median',
                                                "most_frequent"]):

                    if (len(self.__Lcat) != 0):

                        return pd.concat(
                            (pd.DataFrame(self.__imp.transform(df[self.__Lnum]),
                                          columns=self.__Lnum,
                                          index=df.index),
                             df[self.__Lcat].fillna(self.categorical_strategy)
                             ),
                            axis=1)[df.columns]

                    else:

                        return pd.DataFrame(
                            self.__imp.transform(df[self.__Lnum]),
                            columns=self.__Lnum,
                            index=df.index
                        )

                elif ((type(self.numerical_strategy) == int) | (type(self.numerical_strategy) == float)):

                    if (len(self.__Lcat) != 0):

                        return pd.concat(
                            (df[self.__Lnum].fillna(self.numerical_strategy),
                             df[self.__Lcat].fillna(self.categorical_strategy)
                             ),
                            axis=1)[df.columns]

                    else:

                        return df[self.__Lnum].fillna(self.numerical_strategy)

        else:

            raise ValueError("Call fit or fit_transform function before")
