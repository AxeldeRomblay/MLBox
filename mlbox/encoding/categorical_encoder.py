# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings

import os

from tensorflow.keras.layers.core import Dense, Reshape, Dropout
from tensorflow.keras.layers.embeddings import Embedding
from tensorflow.keras.layers import concatenate, Input
from tensorflow.keras.models import Model


class Categorical_encoder():
    """Encodes categorical features.

    Several strategies are possible (supervised or not). Works for both
    classification and regression tasks.

    Parameters
    ----------
    strategy : str, default = "label_encoding"
        The strategy to encode categorical features.
        Available strategies = {"label_encoding", "dummification",
        "random_projection", entity_embedding"}
    verbose : bool, default = False
        Verbose mode. Useful for entity embedding strategy.

    """

    def __init__(self, strategy='label_encoding', verbose=False):
        """Init method for class Categorical_encoder()."""
        self.strategy = strategy
        self.verbose = verbose
        self.__Lcat = []
        self.__Lnum = []
        self.__Enc = dict()
        self.__K = dict()
        self.__weights = None
        self.__fitOK = False

    def get_params(self, deep=True):
        """Get param that can be defined by the user.

        Get strategy parameters and verbose parameters

        Parameters
        ----------
        strategy : str, default = "label_encoding"
            The strategy to encode categorical features.
            Available strategies = {"label_encoding", "dummification",
            "random_projection", entity_embedding"}
        verbose : bool, default = False
            Verbose mode. Useful for entity embedding strategy.

        Returns
        -------
        dict : dictionary
            Dictionary that contains strategy and verbose parameters.

        """
        dict = {'strategy': self.strategy,
                'verbose': self.verbose}
        return dict

    def set_params(self, **params):
        """Set param method for Categorical encoder.

        Set strategy parameters and verbose parameters

        Parameters
        ----------
        strategy : str, default = "label_encoding"
            The strategy to encode categorical features.
            Available strategies = {"label_encoding", "dummification",
            "random_projection", entity_embedding"}
        verbose : bool, default = False
            Verbose mode. Useful for entity embedding strategy.

        """
        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter(s) for encoder "
                              "Categorical_encoder. Parameter(s) IGNORED. "
                              "Check the list of available parameters with "
                              "`encoder.get_params().keys()`")
            else:
                setattr(self, k, v)

    def fit(self, df_train, y_train):
        """Fit Categorical Encoder.

        Encode categorical variable of a dataframe
        following strategy parameters.

        Parameters
        ----------
        df_train : pandas.Dataframe of shape = (n_train, n_features).
            The training dataset with numerical and categorical features.
            NA values are allowed.
        y_train : pandas.Series of shape = (n_train, ).
            The target for classification or regression tasks.

        Returns
        -------
        object
            self

        """
        self.__Lcat = df_train.dtypes[df_train.dtypes == 'object'].index
        self.__Lnum = df_train.dtypes[df_train.dtypes != 'object'].index

        if (len(self.__Lcat) == 0):
            self.__fitOK = True

        else:

            #################################################
            #                Label Encoding
            #################################################

            if (self.strategy == 'label_encoding'):

                for col in self.__Lcat:

                    d = dict()
                    levels = list(df_train[col].unique())
                    nan = False

                    if np.NaN in levels:
                        nan = True
                        levels.remove(np.NaN)

                    for enc, level in enumerate([np.NaN]*nan + sorted(levels)):
                        d[level] = enc  # TODO: Optimize loop?

                    self.__Enc[col] = d

                self.__fitOK = True

            #################################################
            #                Dummification
            #################################################

            elif (self.strategy == 'dummification'):

                for col in self.__Lcat:
                    # TODO: Optimize?
                    self.__Enc[col] = list(df_train[col].dropna().unique())

                self.__fitOK = True

            #################################################
            #                Entity Embedding
            #################################################

            elif (self.strategy == 'entity_embedding'):

                # Parameters
                A = 10   # 15 : more complex
                B = 5    # 2 or 3 : more complex

                # computing interactions
                self.__K = {}
                for col in self.__Lcat:
                    exp_ = np.exp(-df_train[col].nunique() * 0.05)
                    self.__K[col] = np.int(5 * (1 - exp_) + 1)

                sum_ = sum([1. * np.log(k) for k in self.__K.values()])
                # TODO: Add reference for this formula?

                # Number of neurons for layer 1 and 2
                n_layer1 = min(1000,
                               int(A * (len(self.__K) ** 0.5) * sum_ + 1))
                n_layer2 = int(n_layer1 / B) + 2

                # Dropouts
                dropout1 = 0.1
                dropout2 = 0.1

                # Learning parameters
                epochs = 20  # 25 : more iterations
                batch_size = 128  # 256 : gradient more stable

                # Creating the neural network

                embeddings = []
                inputs = []

                for col in self.__Lcat:

                    d = dict()
                    levels = list(df_train[col].unique())
                    nan = False

                    if np.NaN in levels:
                        nan = True
                        levels.remove(np.NaN)

                    for enc, level in enumerate([np.NaN]*nan + sorted(levels)):
                        d[level] = enc  # TODO: Optimize loop?

                    self.__Enc[col] = d

                    var = Input(shape=(1,))
                    inputs.append(var)

                    emb = Embedding(input_dim=len(self.__Enc[col]),
                                    output_dim=self.__K[col],
                                    input_length=1)(var)
                    emb = Reshape(target_shape=(self.__K[col],))(emb)

                    embeddings.append(emb)

                if (len(self.__Lcat) > 1):
                    emb_layer = concatenate(embeddings)
                else:
                    emb_layer = embeddings[0]

                lay1 = Dense(n_layer1,
                             kernel_initializer='uniform',
                             activation='relu')(emb_layer)
                lay1 = Dropout(dropout1)(lay1)

                lay2 = Dense(n_layer2,
                             kernel_initializer='uniform',
                             activation='relu')(lay1)
                lay2 = Dropout(dropout2)(lay2)

                # Learning the weights

                if ((y_train.dtype == object) | (y_train.dtype == 'int')):

                    # Classification
                    if (y_train.nunique() == 2):

                        outputs = Dense(1,
                                        kernel_initializer='normal',
                                        activation='sigmoid')(lay2)

                        model = Model(inputs=inputs, outputs=outputs)
                        model.compile(loss='binary_crossentropy',
                                      optimizer='adam')
                        model.fit(
                            [df_train[col].apply(lambda x: self.__Enc[col][x]).values
                             for col in self.__Lcat],
                            pd.get_dummies(y_train,
                                           drop_first=True).astype(int).values,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=int(self.verbose)
                        )

                    else:

                        outputs = Dense(y_train.nunique(),
                                        kernel_initializer='normal',
                                        activation='softmax')(lay2)

                        model = Model(inputs=inputs, outputs=outputs)
                        model.compile(loss='binary_crossentropy',
                                      optimizer='adam')
                        model.fit(
                            [df_train[col].apply(lambda x: self.__Enc[col][x]).values
                             for col in self.__Lcat],
                            pd.get_dummies(y_train,
                                           drop_first=False).astype(int).values,
                            epochs=epochs,
                            batch_size=batch_size,
                            verbose=int(self.verbose)
                        )

                else:

                    # Regression
                    outputs = Dense(1, kernel_initializer='normal')(lay2)
                    model = Model(inputs=inputs, outputs=outputs)
                    model.compile(loss='mean_squared_error', optimizer='adam')
                    model.fit(
                        [df_train[col].apply(lambda x: self.__Enc[col][x]).values
                         for col in self.__Lcat],
                        y_train.values,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=int(self.verbose)
                    )

                self.__weights = model.get_weights()

                self.__fitOK = True

            #################################################
            #                Random Projection
            #################################################

            elif(self.strategy == 'random_projection'):

                for col in self.__Lcat:

                    exp_ = np.exp(-df_train[col].nunique() * 0.05)
                    # TODO: Add reference to formula used here below?
                    self.__K[col] = np.int(5 * (1 - exp_)) + 1

                    d = dict()
                    levels = list(df_train[col].unique())
                    nan = False

                    if np.NaN in levels:
                        nan = True
                        levels.remove(np.NaN)

                    for k in range(self.__K[col]):

                        if (k == 0):
                            levels = sorted(levels)

                        else:
                            np.random.seed(k)
                            np.random.shuffle(levels)

                        for enc, level in enumerate([np.NaN] * nan + levels):
                            if(k == 0):
                                d[level] = [enc]
                            else:
                                d[level] = d[level] + [enc]

                    self.__Enc[col] = d

                self.__fitOK = True

            else:
                raise ValueError("Categorical encoding strategy is not valid")

        return self

    def fit_transform(self, df_train, y_train):
        """Fits Categorical Encoder and transforms the dataset.

        Fit categorical encoder following strategy parameter and transform the
        dataset df_train.

        Parameters
        ----------
        df_train : pandas.Dataframe of shape = (n_train, n_features)
            The training dataset with numerical and categorical features.
            NA values are allowed.
        y_train : pandas.Series of shape = (n_train, ).
            The target for classification or regression tasks.

        Returns
        -------
        pandas.Dataframe of shape = (n_train, n_features)
            Training dataset with numerical and encoded categorical features.

        """
        self.fit(df_train, y_train)

        return self.transform(df_train)

    def transform(self, df):
        """Transform categorical variable of df dataset.

        Transform df DataFrame encoding categorical features with the strategy
        parameter if self.__fitOK is set to True.

        Parameters
        ----------
        df : pandas.Dataframe of shape = (n_train, n_features)
            The training dataset with numerical and categorical features.
            NA values are allowed.

        Returns
        -------
        pandas.Dataframe of shape = (n_train, n_features)
            The dataset with numerical and encoded categorical features.

        """
        if self.__fitOK:

            if len(self.__Lcat) == 0:
                return df[self.__Lnum]

            else:

                #################################################
                #                Label Encoding
                #################################################

                if (self.strategy == 'label_encoding'):

                    for col in self.__Lcat:

                        # Handling unknown levels
                        unknown_levels = list(set(df[col].values)
                                              - set(self.__Enc[col].keys()))

                        if (len(unknown_levels) != 0):

                            new_enc = len(self.__Enc[col])

                            for unknown_level in unknown_levels:

                                d = self.__Enc[col]
                                # TODO: make sure no collisions introduced
                                d[unknown_level] = new_enc
                                self.__Enc[col] = d

                    if (len(self.__Lnum) == 0):
                        return pd.concat(
                            [pd.DataFrame(
                                df[col].apply(lambda x: self.__Enc[col][x]).values,
                                columns=[col], index=df.index
                                         ) for col in self.__Lcat],
                            axis=1)[df.columns]
                    else:
                        return pd.concat(
                            [df[self.__Lnum]]
                            + [pd.DataFrame(
                                df[col].apply(lambda x: self.__Enc[col][x]).values,
                                columns=[col],
                                index=df.index
                                ) for col in self.__Lcat],
                            axis=1)[df.columns]

                #################################################
                #                 Dummification
                #################################################

                elif (self.strategy == 'dummification'):

                    sub_var = []
                    missing_var = []

                    for col in self.__Lcat:

                        # Handling unknown and missing levels
                        unique_levels = set(df[col].values)
                        sub_levels = unique_levels & set(self.__Enc[col])
                        missing_levels = [col + "_" + str(s)
                                          for s in list(set(self.__Enc[col]) - sub_levels)]
                        sub_levels = [col + "_" + str(s)
                                      for s in list(sub_levels)]

                        sub_var = sub_var + sub_levels
                        missing_var = missing_var + missing_levels

                    if (len(missing_var) != 0):

                        return pd.SparseDataFrame(
                            pd.concat(
                                [pd.get_dummies(df,
                                                sparse=True)[list(self.__Lnum)
                                                             + sub_var]]
                                + [pd.DataFrame(np.zeros((df.shape[0],
                                                          len(missing_var))),
                                                columns=missing_var,
                                                index=df.index)],
                                axis=1
                            )[list(self.__Lnum)+sorted(missing_var+sub_var)])

                    else:

                        return pd.get_dummies(df, sparse=True)[list(self.__Lnum) + sorted(sub_var)]

            #################################################
            #               Entity Embedding
            #################################################

                elif (self.strategy == 'entity_embedding'):

                    def get_embeddings(x, col, i):
                        if int(self.__Enc[col][x]) < \
                                np.shape(self.__weights[i])[0]:
                            return self.__weights[i][int(self.__Enc[col][x]), :]
                        return np.mean(self.__weights[i], axis=0)

                    for col in self.__Lcat:

                        # Handling unknown levels
                        unknown_levels = list(set(df[col].values)
                                              - set(self.__Enc[col].keys())
                                              )

                        if (len(unknown_levels) != 0):

                            new_enc = len(self.__Enc[col])

                            for unknown_level in unknown_levels:

                                d = self.__Enc[col]
                                d[unknown_level] = new_enc
                                self.__Enc[col] = d

                    if (len(self.__Lnum) == 0):
                        return pd.concat(
                            [pd.DataFrame(
                                df[col].apply(lambda x: get_embeddings(x, col, i)).tolist(),
                                columns=[col + "_emb" + str(k + 1)
                                         for k in range(self.__K[col])],
                                index=df.index
                            )
                             for i, col in enumerate(self.__Lcat)], axis=1)
                    else:
                        return pd.concat(
                            [df[self.__Lnum]]
                            + [pd.DataFrame(
                                df[col].apply(lambda x: get_embeddings(x, col, i)).tolist(),
                                columns=[col + "_emb" + str(k + 1)
                                         for k in range(self.__K[col])],
                                index=df.index
                            )
                             for i, col in enumerate(self.__Lcat)], axis=1)

            #################################################
            #               Random Projection
            #################################################

                else:

                    for col in self.__Lcat:

                        unknown_levels = list(set(df[col].values)
                                              - set(self.__Enc[col].keys())
                                              )

                        if (len(unknown_levels) != 0):

                            new_enc = len(self.__Enc[col])

                            for unknown_level in unknown_levels:

                                d = self.__Enc[col]
                                d[unknown_level] = [new_enc
                                                    for _ in range(self.__K[col])]
                                self.__Enc[col] = d

                    if (len(self.__Lnum) == 0):
                        return pd.concat(
                            [pd.DataFrame(
                                df[col].apply(lambda x: self.__Enc[col][x]).tolist(),
                                columns=[col + "_proj" + str(k + 1)
                                         for k in range(self.__K[col])],
                                index=df.index
                            ) for col in self.__Lcat], axis=1)
                    else:
                        return pd.concat(
                            [df[self.__Lnum]]
                            + [pd.DataFrame(
                                df[col].apply(lambda x: self.__Enc[col][x]).tolist(),
                                columns=[col + "_proj" + str(k + 1)
                                         for k in range(self.__K[col])],
                                index=df.index) for col in self.__Lcat], axis=1)

        else:

            raise ValueError("Call fit or fit_transform function before")
