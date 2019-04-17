# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings
import time
import sys
import copy

from hyperopt import fmin, hp, tpe
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer

from .encoding.target_encoder import Target_encoder
from ..prediction.predictor import create_pipeline

# TODO : encode target ici ou dans les etapes du pipe !!!!

def get_metric(scoring, task):

    metric = {}

    if (task == "classification"):

        if (scoring is None):
            scoring = 'roc_auc'

        if (scoring == 'roc_auc'):
            metric[scoring] = make_scorer(lambda y_true, y_pred:
                                          roc_auc_score(pd.get_dummies(y_true), y_pred),
                                          greater_is_better=True,
                                          needs_proba=True)

        else:
            if (type(scoring) == str):

                Lscoring = ["accuracy",
                            "roc_auc",
                            "f1",
                            "log_loss",
                            "precision",
                            "recall"]

                if (scoring in Lscoring):

                    metric[scoring] = scoring

                else:
                    warnings.warn("Invalid scoring metric for classification. "
                                  "'log_loss' is used instead. "
                                  "Available predefined scoring metrics : "
                                  +str(Lscoring))

                    metric["log_loss"] = 'log_loss'

            else:
                metric["custom"] = scoring

    else:

        if (scoring is None):
            scoring = 'mape'

        if (scoring == 'mape'):
            metric[scoring] = make_scorer(lambda y_true, y_pred: 100*np.sum(np.abs(y_true-y_pred)/y_true)/len(y_true),
                                          greater_is_better=False,
                                          needs_proba=False)

        else:

            if (type(scoring) == str):

                Lscoring = ["mape",
                            "mean_absolute_error",
                            "mean_squared_error",
                            "median_absolute_error",
                            "r2"]

                if (scoring in Lscoring):

                    metric[scoring] = scoring

                else:

                    warnings.warn("Invalid scoring metric for regression. "
                                  "'mean_squared_error' is used instead. "
                                  "Available predefined scoring metrics : "
                                  + str(Lscoring))

                    metric["mean_squared_error"] = 'mean_squared_error'

            else:
                metric["custom"] = scoring

    return metric

class Optimiser():

    """Optimises hyper-parameters of the whole Pipeline.

    - NA encoder (missing values encoder)
    - CA encoder (categorical features encoder)
    - Feature selector (OPTIONAL)
    - Stacking estimator - feature engineer (OPTIONAL)
    - Estimator (classifier or regressor)

    Works for both regression and classification (multiclass or binary) tasks.

    Parameters
    ----------
    scoring : str, callable or None. default: None
        A string or a scorer callable object.

        If None, "log_loss" is used for classification and
        "mean_squared_error" for regression

        Available scorings for classification : {"accuracy","roc_auc", "f1",
        "log_loss", "precision", "recall"}

        Available scorings for regression : {"mean_absolute_error",
        "mean_squared_error","median_absolute_error","r2"}

    n_folds : int, default = 2
        The number of folds for cross validation (stratified for classification)

    random_state : int, default = 1
        Pseudo-random number generator state used for shuffling

    to_path : str, default = "save"
        Name of the folder where models are saved

    verbose : bool, default = True
        Verbose mode
    """

    def __init__(self, scoring=None,
                 n_folds=2,
                 random_state=1,
                 to_path="save",
                 verbose=True):

        self.scoring = scoring
        self.n_folds = n_folds
        self.random_state = random_state
        self.to_path = to_path
        self.verbose = verbose

        warnings.warn("Optimiser will save all your fitted models into directory '"
                      +str(self.to_path)+"/joblib'. Please clear it regularly.")        
        
    def get_params(self, deep=True):

        return {'scoring': self.scoring,
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'to_path': self.to_path,
                'verbose': self.verbose}

    def set_params(self, **params):

        self.__fitOK = False

        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for optimiser Optimiser. "
                              "Parameter IGNORED. Check the list of available "
                              "parameters with `optimiser.get_params().keys()`")
            else:
                setattr(self, k, v)

    def evaluate(self, data, params={}):

        """Evaluates the data.

        Evaluates the data with a given scoring function and given hyper-parameters
        of the whole pipeline. If no parameters are set, default configuration for
        each step is evaluated : no feature selection is applied and no meta features are
        created.

        Parameters
        ----------
        data : dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'target' : pandas serie for the target on train set

        params : dict, default = {}
            Hyper-parameters dictionary for the whole pipeline with keys 'encoding' and 'modeling'.

            - params["encoding"] is a dictionary with items :

                - key : 'missing_values' and value : parameter dictionary for 'NA_encoder' object
                - key : 'categorical features' and value : parameter dictionary for 'Categorical_encoder' object

            - params["modeling"] is a dictionary with items :

                - key : 'feature_selection' and value : parameter dictionary for 'NA_encoder' object
                - key : 'stacking_layeri', i=1...n and value : parameter dictionary for 'StackingClassifier' or 'StackingRegressor' object
                - key : 'estimation' and value : parameter dictionary for 'Classifier' or 'Regressor' object

        Returns
        -------
        float.
            The score. The higher the better.
            Positive for a score and negative for a loss.

        Examples
        --------
        >>> from mlbox.optimisation import *
        >>> from sklearn.datasets import load_boston
        >>> #load data
        >>> dataset = load_boston()
        >>> #evaluating the pipeline
        >>> opt = Optimiser()
        >>> params = {
        ...
        ...     "encoding" : {
        ...
        ...         "missing_values" : {
        ...             "numerical_strategy" : 0,
        ...             "categorical_strategy" : "NULL"
        ...             },
        ...
        ...           "categorical_features" : {
        ...                "strategy": "random_projection"
        ...                }
        ...            },
        ...
        ...        "modeling" : {
        ...
        ...            "feature_selection" : {
        ...                "strategy" : "variance",
        ...               "threshold" : 0.2
        ...                },
        ...
        ...            "stacking_layer1" : {
        ...                "n_folds" : 6,
        ...                "verbose" : False
        ...                },
        ...
        ...            "estimation" : {
        ...                "strategy" : "LightGBM",
        ...                "colsample_bytree" : 0.6
        ...                }
        ...            }
        ...        }
        >>> data = {"train" : pd.DataFrame(dataset.data), "target" : pd.Series(dataset.target)}
        >>> opt.evaluate(data, params)
        """

        # Checking the task

        te = Target_encoder()
        te.detect_task(data["target"])
        task = te.get_task()

        # creating cross validation object

        if (task == "classification"):

            counts = data['target'].value_counts()
            classes_to_drop = counts[counts < self.n_folds].index
            mask_to_drop = data['target'].apply(lambda x: x in classes_to_drop)
            indexes_to_drop = data['target'][mask_to_drop].index

            cv = StratifiedKFold(n_splits=self.n_folds,
                                 shuffle=True,
                                 random_state=self.random_state)

        else:

            indexes_to_drop = []
            cv = KFold(n_splits=self.n_folds,
                       shuffle=True,
                       random_state=self.random_state)

        # metrics

        metric = get_metric(self.scoring, task)

        # Creating the Pipeline

        pp = create_pipeline(params, task, self.to_path)

        # Fitting the Pipeline

        start_time = time.time()

        if (self.verbose):
            print("=" * 50 + " TESTING HYPER-PARAMETERS ... " + "=" * 50)
            print("")
            print("> ENCODING: - missing values: " +
                  str(pp.named_steps["missing_values"].get_params()))
            print("")
            print("            - categorical features: " +
                  str(pp.named_steps["categorical_features"].get_params()))
            print("")
            print("")

            sys.stdout.write("> MODELING: ")
            try:
                sys.stdout.write("- feature selection: " +
                                 str(pp.named_steps["feature_selection"].get_params()))
                print("")
                print("")
                sys.stdout.write("            ")

            except:
                pass

            try:
                lay = 0

                for k in np.sort(list(pp.named_steps.keys())):

                    if (k.startswith("stacking_layer")):

                        lay = lay + 1
                        stck_params = pp.named_steps["stacking_layer"+str(lay)]\
                            .get_params().copy()

                        p = {k: stck_params[k] for k in stck_params.keys()
                                if k not in ["level_estimator",
                                            "verbose",
                                            "base_estimators"]}

                        p["base_estimators"] = [est.get_params() for
                                                est in stck_params['base_estimators']]

                        sys.stdout.write("- stacking layer " + str(lay) + ": " + str(p))
                        print("")
                        print("")
                        sys.stdout.write("            ")

            except:
                pass

            est = pp.named_steps["estimation"]

            sys.stdout.write("- estimation: " + str(dict(
                list(est.get_params().items())
                + list(est.get_estimator().get_params().items())
            )))

            print("")
            print("")

            try:

                # Computing the mean cross validation score across the folds
                scores = cross_val_score(estimator=pp,
                                         X=data['train'].drop(indexes_to_drop),
                                         y=te.fit_transform(data['target'].drop(indexes_to_drop)),
                                         scoring=list(metric.items())[0][1],
                                         cv=cv)
                score = np.mean(scores)

            except:

                scores = [-np.inf for _ in range(self.n_folds)]
                score = -np.inf
                warnings.warn("An error occurred while computing the cross "
                              "validation mean score. Check the parameter values "
                              "and your scoring function.")

        ##########################################
        #             Reporting scores
        ##########################################

        out = " ("

        for i, s in enumerate(scores[:-1]):
            out = out + "fold " + str(i + 1) + " = " + str(s) + ", "

        if (self.verbose):
            print("")
            print("MEAN SCORE : " + list(metric.keys())[0] + " = " + str(score))
            print("VARIANCE : " + str(np.std(scores))
                  + out + "fold " + str(i + 2) + " = " + str(scores[-1]) + ")")
            print("CPU time: %s seconds" % (time.time() - start_time))
            print("")

        return score


    def run(self, data, space = {}, max_evals=1):

        """Optimises the Pipeline.

        Optimises hyper-parameters of the whole Pipeline with a given scoring
        function. Algorithm used to optimize : Tree Parzen Estimator.

        Try to avoid dependent parameters and to set one feature
        selection strategy and one estimator strategy at a time.

        Parameters
        ----------
        data : dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'target' : pandas serie for the target on train set

        space : dict, default = {}
            Hyper-parameters space for the whole pipeline with keys 'encoding' and 'modeling'.

            - params["encoding"] is a dictionary with items :

                - key : 'missing_values' and value : parameter dictionary for 'NA_encoder' object
                - key : 'categorical features' and value : parameter dictionary for 'Categorical_encoder' object

            - params["modeling"] is a dictionary with items :

                - key : 'feature_selection' and value : parameter dictionary for 'NA_encoder' object
                - key : 'stacking_layeri', i=1...n and value : parameter dictionary for 'StackingClassifier' or 'StackingRegressor' object
                - key : 'estimation' and value : parameter dictionary for 'Classifier' or 'Regressor' object

        max_evals : int, default = 1.
            Number of iterations.
            To evaluate a configuration, max_evals = 1
            For an accurate optimal hyper-parameter, max_evals = 40.

        Returns
        -------
        dict.
            The optimal hyper-parameter dictionary.

        Examples
        --------
        >>> from mlbox.optimisation import *
        >>> from sklearn.datasets import load_boston
        >>> #load data
        >>> dataset = load_boston()
        >>> #evaluating the pipeline
        >>> opt = Optimiser()
        >>> space = {
        ...
        ...     "encoding" : {
        ...
        ...         "missing_values" : {
        ...             "numerical_strategy" : {0, "mean"},
        ...             "categorical_strategy" : "NULL"
        ...             },
        ...
        ...           "categorical_features" : {
        ...                "strategy": {"random_projection", "label_encoding"}
        ...                }
        ...            },
        ...
        ...        "modeling" : {
        ...
        ...            "feature_selection" : {
        ...                "strategy" : {"variance", "l1"},
        ...               "threshold" : (0.2, 0.3)
        ...                },
        ...
        ...            "stacking_layer1" : {
        ...                "n_folds" : {6},
        ...                "verbose" : {False}
        ...                },
        ...
        ...            "estimation" : {
        ...                "strategy" : {"LightGBM", "XGBoost"},
        ...                "colsample_bytree" : (0.6, 0.9)
        ...                }
        ...            }
        ...        }
        >>> data = {"train" : pd.DataFrame(dataset.data), "target" : pd.Series(dataset.target)}
        >>> opt.run(data, space, max_evals=2)
        """

        start_time = time.time()

        if (self.verbose):
            print("")
            print("STEP 5 - tuning/evaluating the pipeline ...")
            print("")
            print("")

        # checks

        if (type(space) != dict):
            raise ValueError("Parameter space must be a dictionary with keys 'encoding' and 'modeling'. Please read the docs.")

        # Evaluating

        if (len(space) == 0):
            self.evaluate(data)

        # Tuning

        else:

            # Creating a correct space for hyperopt

            K = ["encoding", "modeling"]
            hyper_space = copy.deepcopy(space)

            for step in K:

                if (step in space.keys()):

                    if (type(space[step]) != dict):
                        raise ValueError("Parameter space for " + step + " must be a dictionary. Please read the docs.")

                    else:

                        for k1 in space[step]:

                            for k2 in space[step][k1]:

                                v = space[step][k1][k2]

                                if (type(v) == set):
                                    hyper_space[step][k1][k2] = hp.choice(k1 + "__" + k2, list(v))

                                elif ((type(v) == tuple) | (type(v) == list)):
                                    hyper_space[step][k1][k2] = hp.uniform(k1 + "__" + k2, float(min(v)), float(max(v)))

                                else:
                                    raise ValueError(
                                        "Hyper-parameter values must be either tuples/lists with floats or sets. Please read the docs.")

            # Launching optimisation

            # TODO : auto optim (deap) + better memory

            hyperopt_objective = lambda params: -self.evaluate(data, params)

            best = fmin(hyperopt_objective,
                        space=hyper_space,
                        algo=tpe.suggest,
                        max_evals=max_evals)

            # formatting

            best_params = {"encoding" : {},
                           "modeling" : {}
                           }

            for (k, v) in list(best.items()):

                (k1, k2) = k.split("__")

                if ((k1 == "missing_values") | (k1 == "categorical_features")):
                    step = "encoding"

                elif((k1 == "feature_selection") | (k1.startswith("stacking_layer")) | (k1 == "estimation")):
                    step = "modeling"

                else:
                    step = None

                if (step is not None):

                    range = space[step][k1][k2]

                    if (type(range) == set):
                        value = list(range)[v]
                    else:
                        value = v

                    if (k1 in best_params[step]):

                        best_params[step][k1].update({k2 : value})

                    else:
                        best_params[step][k1] = {k2 : value}

            # Displaying best_params

            if (self.verbose):
                print("")
                print("-" * 50 + "  BEST HYPER-PARAMETERS  " + "-" * 50)
                print("")
                print(best_params)
                print("")
                print("CPU time for STEP 5: %s seconds" % np.round((time.time() - start_time), 2))
                print("")

            return best_params
