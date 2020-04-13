# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np
import os
import sys
import warnings
import time
import operator
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from ..optimisation.encoding.target_encoder import TargetEncoder
from ..optimisation.encoding.nan_encoder import NanEncoder
from ..optimisation.encoding.categorical_encoder import CategoricalEncoder
from ..optimisation.modeling.feature_selector import FeatureSelector
from ..optimisation.modeling.stacking_estimator import StackingEstimator
from ..optimisation.modeling.estimator import Estimator


def create_pipeline(params, task, to_path):

    ##########
    # CREATION
    ##########

    params_pipe = {}
    pipe = []

    if (type(params) != dict):
        raise ValueError("Parameter space must be a dictionary with keys 'encoding' and 'modeling'. Please read the docs.")

    else:

        # ENCODING

        # missing values
        pipe.append(("missing_values", NanEncoder()))

        # categorical features
        pipe.append(("categorical_features", CategoricalEncoder()))

        if ("encoding" in params.keys()):

            if (type(params["encoding"]) != dict):
                raise ValueError("Parameter space for 'encoding' must be a dictionary. Please read the docs.")

            else:

                for k,v in list(params["encoding"].items()):

                    for v1, v2 in list(v.items()):
                        params_pipe[k + "__" + v1] = v2

        else:
            warnings.warn("No encoding strategy is specified. Default configuration is used.")

        # MODELING

        lay = 0

        if ("modeling" in params.keys()):

            if (type(params["modeling"]) != dict):
                raise ValueError("Parameter space for 'modeling' must be a dictionary. Please read the docs.")

            else:

                # feature selection
                if ("feature_selection" in params["modeling"]):

                    if (task == "classification"):
                        pipe.append(("feature_selection", Clf_feature_selector()))

                    else:
                        pipe.append(("feature_selection", Reg_feature_selector()))

                # stacking

                for k in np.sort(list(params["modeling"].keys())):

                    if (k.startswith("stacking_layer")):
                        lay = lay + 1

                        if (task == "classification"):
                            pipe.append(("stacking_layer"+str(lay), StackingClassifier()))

                        else:
                            pipe.append(("stacking_layer" + str(lay), StackingRegressor()))

                for k,v in list(params["modeling"].items()):

                    for v1, v2 in list(v.items()):
                        params_pipe[k + "__" + v1] = v2

        else:
            warnings.warn("No modeling strategy is specified. Default configuration is used.")

        # estimator
        if (task == "classification"):
            pipe.append(("estimation", Classifier()))
        else:
            pipe.append(("estimation", Regressor()))

        ########
        # CACHE
        ########

        cache = False

        if ("categorical_features__strategy" in params_pipe):
            if (params_pipe["categorical_features__strategy"] == "entity_embedding"):
                cache = True

        if ("feature_selection__strategy" in params_pipe):
            if (params_pipe["feature_selection__strategy"] != "variance"):
                cache = True

        if (lay>0):
            cache = True

        ############
        # SET PARAMS
        ############

        if (cache):
            pp = Pipeline(pipe, memory=to_path)
        else:
            pp = Pipeline(pipe)

        try:
            pp.set_params(**params_pipe)

        except:
            raise ValueError("Pipeline cannot be set with these parameters."
                             " Check the name of your stages.")

        return pp

class Predictor():

    """Fits and predicts the target on the test dataset.

    The test dataset must not contain the target values.

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
        self.__pp = None

    def get_params(self, deep=True):

        return {'to_path': self.to_path,
                'verbose': self.verbose
                }

    def set_params(self, **params):

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for predictor Predictor. "
                              "Parameter IGNORED. "
                              "Check the list of available parameters with "
                              "`predictor.get_params().keys()`")
            else:
                setattr(self,k,v)
                
    def __save_feature_importances(self, importance, fig_name="feature_importance.png"):

        """Saves feature importances plot

        Parameters
        ----------
        importance : dict
            Dictionary with features (key) and importances (values)

        fig_name : str, default = "feature_importance.png"
            figure name

        Returns
        -------
        NoneType
            None
        """

        if (len(importance) > 0):

            # Generates plot of feature importances

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
            plt.close()

            # Leak Detection

            leak = sorted(dict(tuples).items(), key=operator.itemgetter(1))[-1]
            if((leak[-1] > 70) & (len(importance) > 1)):
                warnings.warn("WARNING : "
                              + str(leak[0])
                              + " is probably a leak ! "
                                "Please check and delete it...")

        else:
            pass
                

    def __plot_feature_importances(self, importance, top = 10):

        """Plots top 10 feature importances
        
        Parameters
        ----------
        importance : dict
            Dictionary with features (key) and importances (values) 
        
        top : int
            Number of top features to display.
        
        Returns
        -------
        NoneType
            None
        """
        
        if (len(importance) > 0):

            # Plot feature importances
            
            importance_sum = np.sum(list(importance.values()))
            tuples = [(k, np.round(importance[k] * 100. / importance_sum, 2)) 
                      for k in importance]
            tuples = sorted(tuples, key=lambda x: x[1])[-top:]
            labels, values = zip(*tuples)
            plt.figure(figsize=(20, top * 0.3 + 1))

            ylocs = np.arange(len(values))
            plt.barh(ylocs, values, align='center')

            for x, y in zip(values, ylocs):
                plt.text(x + 1, y, x, va='center')

            plt.yticks(ylocs, labels)
            plt.title("Top " + str(top) + " feature importance (%)")
            plt.grid(True)
            plt.show()
            plt.close()

        else:
            pass

    def get_pipeline(self):

        return self.__pp

    def run(self, data, params={}):

        """Fits the model and predicts on the test set.

        Also outputs feature importances and the submission file
        (.png and .csv format).
        
        Parameters
        ----------
        data : dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set



        Returns
        -------
        object
            self.
        """

        if(self.to_path is None):
            raise ValueError("You must specify a path to save your model "
                             "and your predictions")

        else:

            try:
                os.mkdir(self.to_path)
            except OSError:
                pass

            ##########################################
            #               Fitting
            ##########################################

            start_time = time.time()

            if (self.verbose):
                print("")
                print("STEP 6 - fitting the pipeline on training set ...")
                print("")

            # creating the pipeline

            te = Target_encoder()
            te.detect_task(data["target"])
            task = te.get_task()

            self.__pp = create_pipeline(params, task, self.to_path)

            # fitting the Pipeline

            try:

                self.__pp.fit(data['train'], te.fit_transform(data['target']))

                # Feature importances
                try:

                    est = self.__pp.named_steps["estimation"]
                            
                    importance = est.feature_importances()
                    self.__save_feature_importances(importance,
                                                    self.to_path
                                                    + "/"
                                                    + est.get_params()["strategy"]
                                                    + "_feature_importance.png")
                        
                    if(self.verbose):
                        self.__plot_feature_importances(importance, 10)
                        print("")
                        print("> Feature importances dumped into directory : " + self.to_path)

                except:
                    warnings.warn("Unable to get feature importances !")

                if (self.verbose):
                    print("")
                    print("CPU time for STEP 6: %s seconds" % np.round((time.time() - start_time), 2))
                    print("")

            except:
                raise ValueError("Pipeline cannot be fitted.")

            ##########################################
            #               Predicting
            ##########################################

            if (data["test"].shape[0] == 0):
                warnings.warn("You have no test dataset. Cannot predict !")

            else:

                start_time = time.time()

                if (self.verbose):
                    print("")
                    print("STEP 7 - predicting on test set ...")
                    print("")
                    print("")

                try:

                    start_time_pred = time.time()

                    if (self.verbose):
                        sys.stdout.write("predicting ...")

                    # Classification
                    if (task == "classification"):

                        enc = te.get_encoder()

                        pred = pd.DataFrame(self.__pp.predict_proba(data['test']),
                                            columns=enc.inverse_transform(range(len(enc.classes_))),
                                            index=data['test'].index)

                        pred[data['target'].name + "_predicted"] = pred.idxmax(axis=1)

                    # Regression
                    else:

                        pred = pd.DataFrame(self.__pp.predict(data['test']),
                                            columns=[data['target'].name + "_predicted"],
                                            index=data['test'].index)

                    if (self.verbose):
                        sys.stdout.write(" - " + str(np.round(time.time() - start_time_pred, 2)) + " s")
                        print("")

                except:
                    raise ValueError("Can not predict")

                ##########################################
                #               Displaying
                ##########################################

                if(self.verbose):
                    print("")
                    print("> Overview on predictions : ")
                    print("")
                    print(pred.head(10))
                    print("")

                ##########################################
                #           Dumping predictions
                ##########################################

                start_time_dump = time.time()

                if (self.verbose):
                    sys.stdout.write("dumping predictions into directory : "+self.to_path + " ...")

                pred.to_csv(self.to_path
                            + "/"
                            + data['target'].name
                            + "_predictions.csv",
                            index=True)

                if (self.verbose):
                    sys.stdout.write(" - " + str(np.round(time.time() - start_time_dump, 2)) + " s")
                    print("")

                if (self.verbose):
                    print("")
                    print("CPU time for STEP 7: %s seconds" % np.round((time.time() - start_time), 2))
                    print("")

        return self
