# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import numpy as np
import warnings

class IndexEncoder():

    """Encodes index features.
        """

    def __init__(self):

        self.__fitOK = False
        self.__enc = {}
        self.__Lind = []

    def get_params(self, deep=True):

        return {}


    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for encoder IndexEncoder. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`encoder.get_params().keys()`")
            else:
                setattr(self, k, v)


    def detect_type(self, df):

        Lobj = df.dtypes[df.dtypes == 'object'].index
        self.__Lind = []

        for col in Lobj:

            try:

                max_token = df[col].apply(lambda x: len(x.split(" "))).max()

                if (max_token > 2):
                    pass

                else:
                    max_count = df[col].value_count()[0]

                    if(max_count==1):
                        self.__Lind.append(col)
                    else:
                        pass
            except:
                pass

        return self

    def fit(self, df_train, y_train=None):

        """Fits IndexEncoder.

        Parameters
        ----------
        df_train : pandas.Dataframe
            The training dataset with all kind of features.
            NA values are allowed.

        y_train : pandas.Series of shape = (n_train, ).
            The target

        Returns
        -------
        object
            self
        """

        self.__enc = {}

        # detect feature types

        self.detect_type(df_train)

        # fit

        for col in self.__Lind:

            d = dict()
            levels = list(df_train[col].unique())
            nan = False

            if np.NaN in levels:
                nan = True
                levels.remove(np.NaN)

            for enc, level in enumerate([np.NaN] * nan + sorted(levels)):
                d[level] = enc

            self.__enc[col] = d

        self.__fitOK = True

        return self

    def transform(self, df):

        """Transforms the dataset

        Parameters
        ----------
        df : pandas.Dataframe
            A dataset with all kind of features.
            NA values are allowed.

        Returns
        -------
        pandas.Dataframe
            The encoded dataset.
        """

        if (len(self.__Lind) == 0):
            return df

        else:

            L_ind = list(set(df.columns) & set(self.__Lind))
            L_nonind = list(set(df.columns) - set(self.__Lind))

            if (self.__fitOK):

                return