
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause


import pandas as pd
import numpy as np
import os
import pickle
import warnings
import time
import operator
import matplotlib.pyplot as plt
from copy import copy

from sklearn.pipeline import Pipeline

from ..encoding.na_encoder import *
from ..encoding.categorical_encoder import *
from ..model.supervised.classification.feature_selector import *
from ..model.supervised.regression.feature_selector import *
from ..model.supervised.classification.stacking_classifier import *
from ..model.supervised.regression.stacking_regressor import *
from ..model.supervised.classification.classifier import *
from ..model.supervised.regression.regressor import *


class Predictor():

    """
    Predicts the target on the test dataset.


    Parameters
    ----------

    to_path : str, defaut = "save"
        Name of the folder where the model and predictions are saved (python obj and csv format). Must contain target encoder object (for classification task only).

    verbose : bool, defaut = True
        Verbose mode

    """

    def __init__(self, to_path = "save", verbose = True):

        self.to_path = to_path
        self.verbose = verbose

    def get_params(self, deep=True):

        return {'to_path' : self.to_path,
                'verbose' : self.verbose
                }


    def set_params(self,**params):

        self.__fitOK = False

        for k,v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for predictor Predictor. Parameter IGNORED. Check the list of available parameters with `predictor.get_params().keys()`")
            else:
                setattr(self,k,v)


    def __plot_feature_importances(self, importance, fig_name = "feature_importance.png"):


        if(len(importance)>0):

            ### plot feature importances
            tuples = [(k, np.round(importance[k]*100./np.sum(importance.values()),2)) for k in importance]
            tuples = sorted(tuples, key=lambda x: x[1])
            labels, values = zip(*tuples)
            plt.figure(figsize=(20,int(len(importance)*0.3)+1))

            ylocs = np.arange(len(values))
            plt.barh(ylocs, values, align='center')

            for x, y in zip(values, ylocs):
                plt.text(x + 1, y,x, va='center')

            plt.yticks(ylocs,labels)
            plt.title("Feature importance (%)")
            plt.grid(True)
            plt.savefig(fig_name)

            ### leak detection
            leak = sorted(dict(tuples).items(), key=operator.itemgetter(1))[-1]
            if((leak[-1]>70)&(len(importance)>1)):
                warnings.warn("WARNING : "+str(leak[0])+" is probably a leak ! Please check and delete it...")

        else:
            pass



    def fit_predict(self, params, df):


        '''

        Fits the model and saves it. Then predicts on test dataset and outputs the submission file (csv format).


        Parameters
        ----------

        params : dict, defaut = None.
            Hyper-parameters dictionnary for the whole pipeline. If params = None, defaut configuration is evaluated.

            - The keys must respect the following syntax : "enc__param".

            With :
                1/ "enc" = "ne" for na encoder
                2/ "enc" = "ce" for categorical encoder
                3/ "enc" = "fs" for feature selector [OPTIONAL]
                4/ "enc" = "stck"+str(i) to add layer n°i of meta-features (assuming 1 ... i-1 layers are created...) [OPTIONAL]
                5/ "enc" = "est" for the final estimator

            And:
                "param" : a correct associated parameter for each step. (for example : "max_depth" for "enc"="est", "entity_embedding" for "enc"="ce")

            - The values are those of the parameters (for example : 4 for a key = "est__max_depth")


        df : dict, defaut = None
            Dataset dictionnary. Must contain keys "train", "test" and "target" with the train dataset (pandas DataFrame), the test dataset (pandas DataFrame) and the associated
            target (pandas Serie with dtype='float' for a regression or dtype='int' for a classification) resp.


        Returns
        -------

        None

        '''



        if(self.to_path is None):
            raise ValueError("You must specify a path to save your model and your predictions")

        else:

            ne = NA_encoder()
            ce = Categorical_encoder()


            ##########################################################################
            #################### automatically checking the task ####################
            ##########################################################################


            ######################
            ### classification ###

            if(df['target'].dtype=='int'):

                ### estimator ###
                est = Classifier()

                ### feature selection if specified ###
                fs = None
                if(params is not None):
                    for p in params.keys():
                        if(p.startswith("fs__")):
                            fs = Clf_feature_selector()
                        else:
                            pass

                ### stacking if specified ###
                STCK = {}
                if(params is not None):
                    for p in params.keys():
                        if(p.startswith("stck")):
                            STCK[p.split("__")[0]] = StackingClassifier()
                        else:
                            pass


            ##################
            ### regression ###

            elif(df['target'].dtype=='float'):

                ### estimator ###
                est = Regressor()

                 ### feature selection if specified ###
                fs = None
                if(params is not None):
                    for p in params.keys():
                        if(p.startswith("fs__")):
                            fs = Reg_feature_selector()
                        else:
                            pass

                ### stacking if specified ###
                STCK = {}
                if(params is not None):
                    for p in params.keys():
                        if(p.startswith("stck")):
                            STCK[p.split("__")[0]] = StackingRegressor()
                        else:
                            pass


            else:
                raise ValueError("Impossible to determine the task. Please check that your target is encoded.")


            #############################################################
            ##################### creating the pipeline #################
            #############################################################

            pipe = [("ne",ne),("ce",ce)]

            if(fs is not None):
                pipe.append(("fs",fs))
            else:
                pass

            for stck in np.sort(STCK.keys()):
                pipe.append((stck,STCK[stck]))

            pipe.append(("est",est))
            pp = Pipeline(pipe)


            #############################################################
            #################### fitting the pipeline ###################
            #############################################################

            start_time = time.time()

            ### no params : defaut config ###
            if(params is None):
                print("")
                print('No parameters set. Default configuration is tested')
                set_params = True

            else:
                try:
                    pp = pp.set_params(**params)
                    set_params = True
                except:
                    set_params = False

            if(set_params):

                try:
                    if(self.verbose):
                        print("")
                        print("fitting the pipeline...")

                    pp.fit(df['train'], df['target'])

                    if(self.verbose):
                        print("CPU time: %s seconds" % (time.time() - start_time))

                    try:
                        os.mkdir(self.to_path)
                    except OSError:
                        pass


                    ### feature importances ###
                    try:
                        importance = est.feature_importances()
                        self.__plot_feature_importances(importance, self.to_path+"/"+est.get_params()["strategy"]+"_feature_importance.png")
                    except:
                        pass

                except:
                    raise ValueError("Pipeline cannot be fitted")
            else:
                raise ValueError("Pipeline cannot be set with these parameters. Check the name of your stages.")

            ############################################################
            ######################## predicting #######################
            ############################################################

            if (df["test"].shape[0] == 0):
                warnings.warn("You have no test dataset. Cannot predict !")

            else:

                start_time = time.time()

                ######################
                ### classification ###

                if(df['target'].dtype=='int'):

                    try:

                        fhand = open(self.to_path+"/target_encoder.obj", 'r')
                        enc = pickle.load(fhand)
                        fhand.close()

                    except:
                        raise ValueError("Unable to load target encoder from directory : "+self.to_path)

                    try:
                        if(self.verbose):
                            print("")
                            print("predicting...")

                        pred = pd.DataFrame(pp.predict_proba(df['test']),columns = enc.inverse_transform(range(len(enc.classes_))), index = df['test'].index)
                        pred[df['target'].name+"_predicted"] = pred.idxmax(axis=1)
                        try:
                            pred[df['target'].name+"_predicted"] = pred[df['target'].name+"_predicted"].apply(int)
                        except:
                            pass

                    except:
                        raise ValueError("Can not predict")

                ######################
                ### regression ###

                elif(df['target'].dtype=='float'):

                    pred = pd.DataFrame([],columns=[df['target'].name+"_predicted"], index = df['test'].index)

                    try:
                        if(self.verbose):
                            print("")
                            print("predicting...")

                        pred[df['target'].name+"_predicted"] = pp.predict(df['test'])

                    except:
                        raise ValueError("Can not predict")

                else:
                    pass

                if(self.verbose):
                    print("CPU time: %s seconds" % (time.time() - start_time))

                ############################################################
                ######################## Displaying #######################
                ############################################################

                if(self.verbose):
                    print("")
                    print("top 10 predictions :")
                    print("")
                    print(pred.head(10))

                ############################################################
                #################### dumping predictions ###################
                ############################################################

                if(self.verbose):
                    print("")
                    print("dumping predictions into directory : "+self.to_path)

                pred.to_csv(self.to_path+"/"+df['target'].name+"_predictions.csv",index=True)

        return self
