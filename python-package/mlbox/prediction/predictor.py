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

from sklearn.pipeline import Pipeline

from ..encoding.na_encoder import NA_encoder
from ..encoding.categorical_encoder import Categorical_encoder
from ..model.supervised.classification.feature_selector import Clf_feature_selector
from ..model.supervised.regression.feature_selector import Reg_feature_selector
from ..model.supervised.classification.stacking_classifier import StackingClassifier
from ..model.supervised.regression.stacking_regressor import StackingRegressor
from ..model.supervised.classification.classifier import Classifier
from ..model.supervised.regression.regressor import Regressor


class Predictor():

    """
    Predicts the target on the test dataset.


    Parameters
    ----------

    to_path : str, default = "save"
        Name of the folder where feature importances and
        predictions are saved (.png and .csv formats).
        Must contain target encoder object (for classification task only).

    verbose : bool, default = True
        Verbose mode

    """

    def __init__(self, to_path="save", verbose=True):

        self.to_path = to_path
        self.verbose = verbose

    def get_params(self, deep=True):

        return {'to_path': self.to_path,
                'verbose': self.verbose
                }

    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for predictor Predictor. "
                              "Parameter IGNORED. "
                              "Check the list of available parameters with "
                              "`predictor.get_params().keys()`")
            else:
                setattr(self, k, v)

    def __plot_feature_importances(self,
                                   importance,
                                   fig_name="feature_importance.png"):

        """
        Saves feature importances plot

        Parameters
        ----------

        importance : dict
            Dictionary with features (key) and importances (values)

        fig_name : str, default = "feature_importance.png"
            figure name


        Returns
        -------

        None
        """

        if (len(importance) > 0):

            # Plot feature importances

            importance_sum = np.sum(list(importance.values()))
            tuples = [(k, np.round(importance[k] * 100. / importance_sum, 2))
                      for k in importance]
            tuples = sorted(tuples, key=lambda x: x[1])
            labels, values = zip(*tuples)
            plt.figure(figsize=(20, int(len(importance) * 0.3) + 1))

            ylocs = np.arange(len(values))
            plt.barh(ylocs, values, align='center')

            for x, y in zip(values, ylocs):
                plt.text(x + 1, y, x, va='center')

            plt.yticks(ylocs, labels)
            plt.title("Feature importance (%)")
            plt.grid(True)
            plt.savefig(fig_name)

            # Leak Detection

            leak = sorted(dict(tuples).items(), key=operator.itemgetter(1))[-1]
            if((leak[-1] > 70) & (len(importance) > 1)):
                warnings.warn("WARNING : "
                              + str(leak[0])
                              + " is probably a leak ! "
                                "Please check and delete it...")

        else:
            pass

    def fit_predict(self, params, df):


        '''
        Fits the model. Then predicts on test dataset and outputs feature
        importances and the submission file (.png and .csv format).

        
        Parameters
        ----------
        
        params : dict, default = None.
            Hyper-parameters dictionary for the whole pipeline.
            If params = None, default configuration is evaluated.
            
            - The keys must respect the following syntax : "enc__param".
            
            With :
                1/ "enc" = "ne" for na encoder
                2/ "enc" = "ce" for categorical encoder
                3/ "enc" = "fs" for feature selector [OPTIONAL]
                4/ "enc" = "stck"+str(i) to add layer n°i of meta-features
                (assuming 1 ... i-1 layers are created...) [OPTIONAL]
                5/ "enc" = "est" for the final estimator
            
            And:
                "param" : a correct associated parameter for each step.
                (for example : "max_depth" for "enc"="est",
                               "entity_embedding" for "enc"="ce")
            
            - The values are those of the parameters
                (for example : 4 for key = "est__max_depth")
        
        
        df : dict, default = None
            Dataset dictionary. Must contain keys "train", "test"
            and "target" with the train dataset (pandas.DataFrame),
            the test dataset (pandas.DataFrame) and the associated
            target (pandas Serie with dtype='float' for a regression or
            dtype='int' for a classification)
        
        
        Returns
        -------
        
        None
        '''

        if(self.to_path is None):
            raise ValueError("You must specify a path to save your model "
                             "and your predictions")

        else:

            ne = NA_encoder()
            ce = Categorical_encoder()

            ##########################################
            #    Automatically checking the task
            ##########################################

            ##########################################
            #             Classification
            ##########################################

            if (df['target'].dtype == 'int'):

                # Estimator

                est = Classifier()

                # Feature selection if specified

                fs = None
                if(params is not None):
                    for p in params.keys():
                        if(p.startswith("fs__")):
                            fs = Clf_feature_selector()
                        else:
                            pass

                # Stacking if specified

                STCK = {}
                if(params is not None):
                    for p in params.keys():
                        if(p.startswith("stck")):
                            STCK[p.split("__")[0]] = StackingClassifier()
                        else:
                            pass

        ##########################################
        #               Regression
        ##########################################

            elif (df['target'].dtype == 'float'):

                # Estimator

                est = Regressor()

                # Feature selection if specified

                fs = None
                if(params is not None):
                    for p in params.keys():
                        if(p.startswith("fs__")):
                            fs = Reg_feature_selector()
                        else:
                            pass

                # Stacking if specified

                STCK = {}
                if(params is not None):
                    for p in params.keys():
                        if(p.startswith("stck")):
                            STCK[p.split("__")[0]] = StackingRegressor()
                        else:
                            pass

            else:
                raise ValueError("Impossible to determine the task. "
                                 "Please check that your target is encoded.")

            ##########################################
            #          Creating the Pipeline
            ##########################################

            pipe = [("ne", ne), ("ce", ce)]

            # Do we need to cache transformers?

            cache = False

            if (params is not None):
                if("ce__strategy" in params):
                    if(params["ce__strategy"] == "entity_embedding"):
                        cache = True
                    else:
                        pass
                else:
                    pass

            if (fs is not None):
                if ("fs__strategy" in params):
                    if(params["fs__strategy"] != "variance"):
                        cache = True
                    else:
                        pass
            else:
                pass

            if (len(STCK) != 0):
                cache = True
            else:
                pass

            # Pipeline creation

            if (fs is not None):
                pipe.append(("fs", fs))
            else:
                pass

            for stck in np.sort(list(STCK)):
                pipe.append((stck, STCK[stck]))

            pipe.append(("est", est))

            if(cache):
                pp = Pipeline(pipe, memory=self.to_path)
            else:
                pp = Pipeline(pipe)

            ##########################################
            #          Fitting the Pipeline
            ##########################################

            start_time = time.time()

            # No params : default configuration

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
                        print("CPU time: %s seconds"%(time.time() - start_time))

                    try:
                        os.mkdir(self.to_path)
                    except OSError:
                        pass

                    # Feature importances

                    try:
                        importance = est.feature_importances()
                        self.__plot_feature_importances(importance,
                                                        self.to_path
                                                        + "/"
                                                        + est.get_params()["strategy"]
                                                        + "_feature_importance.png")
                    except:
                        warnings.warn("Unable to get feature importances...")

                except:
                    raise ValueError("Pipeline cannot be fitted")
            else:
                raise ValueError("Pipeline cannot be set with these parameters."
                                 " Check the name of your stages.")

            ##########################################
            #               Predicting
            ##########################################

            if (df["test"].shape[0] == 0):
                warnings.warn("You have no test dataset. Cannot predict !")

            else:

                start_time = time.time()

                ##########################################
                #             Classification
                ##########################################

                if (df['target'].dtype == 'int'):

                    try:

                        fhand = open(self.to_path + "/target_encoder.obj", 'rb')
                        enc = pickle.load(fhand)
                        fhand.close()

                    except:
                        raise ValueError("Unable to load target encoder"
                                         " from directory : " + self.to_path)

                    try:
                        if(self.verbose):
                            print("")
                            print("predicting...")

                        pred = pd.DataFrame(pp.predict_proba(df['test']),
                                            columns=enc.inverse_transform(range(len(enc.classes_))),
                                            index=df['test'].index)
                        pred[df['target'].name + "_predicted"] = pred.idxmax(axis=1)  # noqa
                        
                        try:
                            pred[df['target'].name + "_predicted"] = pred[df['target'].name + "_predicted"].apply(int)  # noqa
                        except:
                            pass

                    except:
                        raise ValueError("Can not predict")

                ##########################################
                #               Regression
                ##########################################

                elif (df['target'].dtype == 'float'):

                    pred = pd.DataFrame([],
                                        columns=[df['target'].name + "_predicted"],
                                        index=df['test'].index)

                    try:
                        if(self.verbose):
                            print("")
                            print("predicting...")

                        pred[df['target'].name + "_predicted"] = pp.predict(df['test'])  # noqa

                    except:
                        raise ValueError("Can not predict")

                else:
                    pass

                if(self.verbose):
                    print("CPU time: %s seconds" % (time.time() - start_time))

                ##########################################
                #               Displaying
                ##########################################

                if(self.verbose):
                    print("")
                    print("top 10 predictions :")
                    print("")
                    print(pred.head(10))

                ##########################################
                #           Dumping predictions
                ##########################################

                if(self.verbose):
                    print("")
                    print("dumping predictions into directory : "+self.to_path)

                pred.to_csv(self.to_path
                            + "/"
                            + df['target'].name
                            + "_predictions.csv",
                            index=True)

        return self
