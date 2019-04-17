# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
import warnings

from .target_encoder import TargetEncoder

class TextEncoder():

    """Encodes text features.

        Several strategies are possible (supervised or not). Works for both
        classification and regression tasks.

        Parameters
        ----------
        strategy : str, default = "SVD"
            The strategy to encode text features.
            Available strategies = {"SVD", "LGBM"}

        task : str, default = "auto"
            The task ("classification" or "regression").
        """

    def __init__(self, strategy='SVD', task="auto"):

        self.strategy = strategy
        self.task = task
        self.__detectedtask = None
        self.__fitOK = False
        self.__enc = {}
        self.__Ltext = []

    def get_params(self, deep=True):

        return {'strategy': self.strategy}


    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for encoder Text_encoder. "
                              "Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`encoder.get_params().keys()`")
            else:
                setattr(self, k, v)

    def detect_type(self, df):

        Lcat = df.dtypes[df.dtypes == 'object'].index
        self.__Ltext = []

        for col in Lcat:

            # TODO : tester le value counts

            try:
                max_token = df[col].apply(lambda x: len(x.split(" "))).max()
                if(max_token > 2):
                    self.__Ltext.append(col)
                else:
                    pass
            except:
                pass

        return self

    def fit(self, df_train, y_train):

        """Fits Text Encoder.

        Parameters
        ----------
        df_train : pandas.Dataframe of shape = (n_train, n_features).
            The training dataset with numerical, categorical, text or list features.
            NA values are not allowed.

        y_train : pandas.Series of shape = (n_train, ).
            The target for classification or regression tasks.

        Returns
        -------
        object
            self
        """

        self.__enc = {}

        # detect feature types

        self.detect_type(df_train)

        # detect task if auto

        if(self.task=="auto"):
            te = TargetEncoder()
            te.detect_task(y_train)
            self.__detectedtask = te.get_task()

        elif((self.task=="classification") | (self.task=="regression")):
            self.__detectedtask = self.task

        else:
            raise ValueError("Please set your task : 'classification', 'regression' or 'auto'")

        # fit

        if(len(self.__Ltext)==0):
            self.__fitOK = True

        else:

            for col in self.__Ltext:

                enc1 = HashingVectorizer(n_features=100000, stop_words="english")

                if(self.strategy=="LGBM"):

                    if(self.__detectedtask=="classification"):
                        enc2 = LGBMClassifier(n_estimators=300, colsample_bytree=0.6, max_depth=6, learning_rate=0.05, subsample=0.85, seed=1)
                    else:
                        enc2 = LGBMRegressor(n_estimators=300, colsample_bytree=0.6, max_depth=6, learning_rate=0.05, subsample=0.85, seed=1)

                elif(self.strategy=="SVD"):

                    enc2 = TruncatedSVD(n_components=50, random_state=1)

                else:
                    raise ValueError("Strategy for text encoding is not valid")

                enc = Pipeline([("enc1", enc1), ("enc2", enc2)])
                enc.fit(df_train[col], y)

                self.__enc[col] = enc

            self.__fitOK = True

        return self

    def transform(self, df):

        """Transforms the dataset

        Parameters
        ----------
        df : pandas.Dataframe
            A dataset with numerical, categorical, text or list features.
            NA values are not allowed.

        Returns
        -------
        pandas.Dataframe
            The encoded dataset.
        """

        if(len(self.__Ltext)==0):
            return df

        else:

            L_text = list(set(df.columns) & set(self.__Ltext))
            L_nontext = list(set(df.columns) - set(self.__Ltext))

            if(self.__fitOK):

                if(self.strategy=="LGBM"):

                    if(self.__detectedtask=="classification"):

                        n_classes = {}
                        for col in L_text:
                            n_classes[col] = len(self.__enc[col].named_steps["enc2"].classes_)

                        return pd.concat([pd.DataFrame(self.__enc[col].predict_proba(df[col])[:,1:],
                                                       index=df.index,
                                                       columns=[col+"_score"+str(i)
                                                                for i in range(1, n_classes[col])
                                                               ]) for col in L_text] +
                                          [pd.DataFrame(df[col].apply(len).values,
                                                        columns=[col+"_length"],
                                                        index=df.index) for col in L_text] +
                                         [df[L_nontext]],
                                         axis=1)
                    else:

                        return pd.concat([pd.DataFrame(self.__enc[col].predict(df[col]),
                                                       index=df.index,
                                                       columns=[col+"_score"])
                                          for col in L_text] +
                                          [pd.DataFrame(df[col].apply(len).values,
                                                        columns=[col+"_length"],
                                                        index=df.index) for col in L_text] +
                                         [df[L_nontext]],
                                         axis=1)

                else:

                    n_comp = {}
                    for col in L_text:
                        n_comp[col] = self.__enc[col].named_steps["enc2"].n_components

                    return pd.concat([pd.DataFrame(self.__enc[col].transform(df[col]),
                                                   index=df.index,
                                                   columns=[col+"_proj"+str(i) for i in range(1,n_comp[col]+1)])
                                      for col in L_text] +
                                      [pd.DataFrame(df[col].apply(len).values,
                                                    columns=[col+"_length"],
                                                    index=df.index)] +
                                     [df[L_nontext]],
                                     axis=1)

            else:

                raise ValueError("Call fit or fit_transform function before")

    def fit_transform(self, df_train, y_train):

        """Fits Text Encoder and transforms the dataset

        Parameters
        ----------
        df_train : pandas.Dataframe
            The training dataset with all kind of features.
            NA values are not allowed.

        y_train : pandas.Series of shape = (n_train, ).
            The target for classification or regression tasks.

        Returns
        -------
        pandas.Dataframe
            The encoded training dataset.
        """

        self.fit(df_train, y_train)
        return self.transform(df_train)