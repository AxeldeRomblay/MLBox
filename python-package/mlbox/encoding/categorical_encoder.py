# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings

import os

with open(os.path.expanduser('~')+'/.keras/keras.json','w') as f:
    new_settings = """{\r\n
    "epsilon": 1e-07,\r\n
    "image_data_format": "channels_last",\n
    "backend": "theano",\r\n
    "floatx": "float32"\r\n
    }"""
    f.write(new_settings)

from keras.layers.core import Dense, Reshape, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import concatenate, Input
from keras.models import Model


class Categorical_encoder():

    """
    Encodes categorical features. Several strategies are possible (supervised or not). Works for both classification and regression tasks.


    Parameters
    ----------

    strategy : string, defaut = "label_encoding"
        The strategy to encode categorical features.
        Available strategies = "label_encoding", "dummification", "random_projection", entity_embedding"

    verbose : boolean, defaut = False
        Verbose mode. Useful for entity embedding strategy.

    """

    def __init__(self, strategy = 'label_encoding', verbose = False):

        self.strategy = strategy    #label_encoding, dummification, random_projection and entity_embedding
        self.verbose = verbose
        self.__Lcat = []
        self.__Lnum = []
        self.__Enc = dict()
        self.__K = dict()
        self.__weights = None
        self.__fitOK = False


    def get_params(self, deep = True):

        return {'strategy' : self.strategy,
                'verbose' : self.verbose}


    def set_params(self,**params):

        self.__fitOK = False

        for k,v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for encoder Categorical_encoder. Parameter IGNORED. Check the list of available parameters with `encoder.get_params().keys()`")
            else:
                setattr(self,k,v)



    def fit(self, df_train, y_train):

        '''

        Fits Categorical Encoder.

        Parameters
        ----------

        df_train : pandas dataframe of shape = (n_train, n_features)
        The train dataset with numerical and categorical features. NA values are allowed.

        y_train : pandas series of shape = (n_train, ).
        The target for classification or regression tasks.


        Returns
        -------
        None

        '''

        self.__Lcat = df_train.dtypes[df_train.dtypes == 'object'].index
        self.__Lnum = df_train.dtypes[df_train.dtypes != 'object'].index

        ###################################################
        ################# Label encoding #################
        ###################################################


        if(self.strategy=='label_encoding'):

            if(len(self.__Lcat)==0):
                pass

            else:
                for col in self.__Lcat:

                    d = dict()
                    for enc,level in enumerate(list(np.sort(df_train[col].unique()))):
                        d[level] = enc  #boucle a optimiser ?
                    self.__Enc[col] = d


            self.__fitOK = True

        ###################################################
        ################# dummification #################
        ###################################################

        elif(self.strategy=='dummification'):

            if(len(self.__Lcat)==0):
                pass

            else:
                for col in self.__Lcat:

                    self.__Enc[col] = list(df_train[col].dropna().unique())  #a optimiser ?

            self.__fitOK = True

        ###################################################
        ################# entity embedding #################
        ###################################################

        elif(self.strategy=='entity_embedding'):

            if(len(self.__Lcat)==0):
                pass

            else:

                ### parameters ###

                A = 10   # 15 : more complex
                B = 5    # 2 or 3 : more complex

                # number of neurons for layer 1 et 2
                n_layer1 = min(1000,int(A*(len(self.__K)**0.5)*sum([1.*np.log(k) for k in self.__K.values()])+1))    # tuning
                n_layer2 = n_layer1/B + 2

                #dropouts
                dropout1 = 0.1
                dropout2 = 0.1

                #learning parameters
                epochs = 20  #25 : more iterations
                batch_size = 128 # 256 : gradient more stable


                ### creating the neural network ###

                embeddings = []
                inputs = []

                for col in self.__Lcat:

                    self.__K[col] = np.int(5*(1-np.exp(-df_train[col].nunique()*0.05)))+1

                    d = dict()

                    for enc,level in enumerate(list(np.sort(df_train[col].unique()))):
                        d[level] = enc

                    self.__Enc[col] = d

                    var = Input(shape=(1,))
                    inputs.append(var)

                    emb = Embedding(input_dim = len(self.__Enc[col]), output_dim = self.__K[col], input_length=1)(var)
                    emb = Reshape(target_shape=(self.__K[col],))(emb)

                    embeddings.append(emb)

                if(len(self.__Lcat)>1):
                    emb_layer = concatenate(embeddings)
                else:
                    emb_layer = embeddings[0]


                lay1 = Dense(n_layer1, kernel_initializer='uniform', activation='relu')(emb_layer)
                lay1 = Dropout(dropout1)(lay1)

                lay2 = Dense(n_layer2, kernel_initializer='uniform', activation='relu')(lay1)
                lay2 = Dropout(dropout2)(lay2)


                ### fitting the weights ###

                if((y_train.dtype==object)|(y_train.dtype=='int')):

                    # classification
                    if(y_train.nunique()==2):

                        outputs = Dense(y_train.nunique()-1, kernel_initializer='normal', activation='sigmoid')(lay2)
                        model = Model(inputs=inputs, outputs = outputs)
                        model.compile(loss='binary_crossentropy', optimizer='adam')
                        model.fit([df_train[col].apply(lambda x: self.__Enc[col][x]).values for col in self.__Lcat], pd.get_dummies(y_train,drop_first=True).astype(int).values, epochs=epochs, batch_size=batch_size, verbose=int(self.verbose))

                    else:

                        outputs = Dense(y_train.nunique(), kernel_initializer='normal', activation='sigmoid')(lay2)
                        model = Model(inputs=inputs, outputs = outputs)
                        model.compile(loss='binary_crossentropy', optimizer='adam')
                        model.fit([df_train[col].apply(lambda x: self.__Enc[col][x]).values for col in self.__Lcat], pd.get_dummies(y_train,drop_first=False).astype(int).values, epochs=epochs, batch_size=batch_size, verbose=int(self.verbose))


                else:

                    # regression
                    outputs = Dense(1, kernel_initializer='normal')(lay2)
                    model = Model(inputs=inputs, outputs = outputs)
                    model.compile(loss='mean_squared_error', optimizer='adam')
                    model.fit([df_train[col].apply(lambda x: self.__Enc[col][x]).values for col in self.__Lcat], y_train.values, epochs=epochs, batch_size=batch_size, verbose=int(self.verbose))


                self.__weights = model.get_weights()

            self.__fitOK = True

        ###################################################
        ################# random projection #################
        ###################################################

        elif(self.strategy=='random_projection'):

            if(len(self.__Lcat)==0):
                pass

            else:
                for col in self.__Lcat:

                    self.__K[col] = np.int(5*(1-np.exp(-df_train[col].nunique()*0.05)))+1

                    d = dict()
                    levels = list(np.sort(df_train[col].unique()))

                    for k in range(self.__K[col]):

                        np.random.seed(k)
                        np.random.shuffle(levels)

                        for enc,level in enumerate(levels):
                            if(k==0):
                                d[level] = [enc]
                            else:
                                d[level] = d[level] + [enc]

                    self.__Enc[col] = d


            self.__fitOK = True


        else:

            raise ValueError("strategy for categorical encoding is not valid")

        return self



    def fit_transform(self, df_train, y_train):

        '''

        Fits Categorical Encoder and transforms the dataset

        Parameters
        ----------

        df_train : pandas dataframe of shape = (n_train, n_features)
        The train dataset with numerical and categorical features. NA values are allowed.

        y_train : pandas series of shape = (n_train, ).
        The target for classification or regression tasks.


        Returns
        -------

        df_train : pandas dataframe of shape = (n_train, n_features)
        The train dataset with numerical and encoded categorical features.

        '''

        self.fit(df_train, y_train)

        return self.transform(df_train)



    def transform(self, df):

        '''

        Transforms the dataset

        Parameters
        ----------

        df : pandas dataframe of shape = (n, n_features)
        The dataset with numerical and categorical features. NA values are allowed.


        Returns
        -------

        df : pandas dataframe of shape = (n, n_features)
        The dataset with numerical and encoded categorical features.

        '''

        if(self.__fitOK):

            if(len(self.__Lcat)==0):
                return df[self.__Lnum]

            else:

                ###################################################
                ################# Label encoding #################
                ###################################################

                if(self.strategy=='label_encoding'):

                    for col in self.__Lcat:

                        ### handling unknown levels ###
                        unknown_levels = list(set(df[col].values) - set(self.__Enc[col].keys()))

                        if(len(unknown_levels)!=0):

                            new_enc = len(self.__Enc[col])

                            for unknown_level in unknown_levels:

                                d = self.__Enc[col]
                                d[unknown_level] = new_enc
                                self.__Enc[col] = d


                    if(len(self.__Lnum)==0):
                        return pd.concat([pd.DataFrame(df[col].apply(lambda x: self.__Enc[col][x]).values,
                                                   columns=[col],index=df.index) for col in self.__Lcat],axis=1)[df.columns]
                    else:
                        return pd.concat([df[self.__Lnum]]+[pd.DataFrame(df[col].apply(lambda x: self.__Enc[col][x]).values,
                                                   columns=[col],index=df.index) for col in self.__Lcat],axis=1)[df.columns]

                ###################################################
                ################# dummification #################
                ###################################################

                elif(self.strategy=='dummification'):

                    sub_var = []
                    missing_var = []

                    for col in self.__Lcat:

                        ### handling unknown and missing levels ###
                        unique_levels = set(df[col].values)
                        sub_levels = unique_levels & set(self.__Enc[col])
                        missing_levels = [col+"_"+ s for s in list(set(self.__Enc[col]) - sub_levels)]
                        sub_levels = [col+"_"+ s for s in list(sub_levels)]

                        sub_var = sub_var + sub_levels
                        missing_var = missing_var + missing_levels


                    if(len(missing_var)!=0):

                        return pd.SparseDataFrame(pd.concat([pd.get_dummies(df,sparse=True)[list(self.__Lnum)+sub_var]]+[pd.DataFrame(np.zeros((df.shape[0],len(missing_var))),columns=missing_var,index=df.index)],axis=1)[list(self.__Lnum)+sorted(missing_var+sub_var)])


                    else:

                        return pd.get_dummies(df,sparse=True)[list(self.__Lnum)+sorted(sub_var)]

                ###################################################
                ################# entity embedding #################
                ###################################################

                elif(self.strategy=='entity_embedding'):

                    for col in self.__Lcat:

                        ### handling unknown levels ###
                        unknown_levels = list(set(df[col].values) - set(self.__Enc[col].keys()))

                        if(len(unknown_levels)!=0):

                            new_enc = len(self.__Enc[col])

                            for unknown_level in unknown_levels:

                                d = self.__Enc[col]
                                d[unknown_level] = new_enc
                                self.__Enc[col] = d


                    if(len(self.__Lnum)==0):
                        return pd.concat([pd.DataFrame(df[col].apply(lambda x: self.__weights[i][int(self.__Enc[col][x]),:] if int(self.__Enc[col][x])<np.shape(self.__weights[i])[0] else np.mean(self.__weights[i],axis=0)).tolist(),columns=[col+"_emb"+str(k+1) for k in range(self.__K[col])], index = df.index)
                                            for i, col in enumerate(self.__Lcat)],axis=1)
                    else:

                        return pd.concat([df[self.__Lnum]]+[pd.DataFrame(df[col].apply(lambda x: self.__weights[i][int(self.__Enc[col][x]),:] if int(self.__Enc[col][x])<np.shape(self.__weights[i])[0] else np.mean(self.__weights[i],axis=0)).tolist(),columns=[col+"_emb"+str(k+1) for k in range(self.__K[col])], index = df.index)
                                                                for i, col in enumerate(self.__Lcat)
                                                            ],axis=1)

                ###################################################
                ################# random projection #################
                ###################################################

                else:

                    for col in self.__Lcat:

                        unknown_levels = list(set(df[col].values) - set(self.__Enc[col].keys()))

                        if(len(unknown_levels)!=0):

                            new_enc = len(self.__Enc[col])

                            for unknown_level in unknown_levels:

                                d = self.__Enc[col]
                                d[unknown_level] = [new_enc for k in range(self.__K[col])]
                                self.__Enc[col] = d

                    if(len(self.__Lnum)==0):
                        return pd.concat([pd.DataFrame(df[col].apply(lambda x: self.__Enc[col][x]).tolist(),
                                                columns=[col+"_proj"+str(k+1) for k in range(self.__K[col])],index=df.index) for col in self.__Lcat],axis=1)
                    else:
                        return pd.concat([df[self.__Lnum]]+[pd.DataFrame(df[col].apply(lambda x: self.__Enc[col][x]).tolist(),
                                                   columns=[col+"_proj"+str(k+1) for k in range(self.__K[col])],index=df.index) for col in self.__Lcat],axis=1)

        else:

            raise ValueError("call fit or fit_transform function before")

