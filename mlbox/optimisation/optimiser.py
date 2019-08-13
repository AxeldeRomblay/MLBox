# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings
import time

from hyperopt import fmin, hp, tpe
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, make_scorer

from ..encoding.na_encoder import NA_encoder
from ..encoding.categorical_encoder import Categorical_encoder
from ..model.classification.feature_selector import Clf_feature_selector
from ..model.regression.feature_selector import Reg_feature_selector
from ..model.classification.stacking_classifier import StackingClassifier
from ..model.regression.stacking_regressor import StackingRegressor
from ..model.classification.classifier import Classifier
from ..model.regression.regressor import Regressor


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

        If None, "neg_log_loss" is used for classification and
        "neg_mean_squared_error" for regression

        Available scorings for classification : {"accuracy","roc_auc", "f1",
        "neg_log_loss", "precision", "recall"}

        Available scorings for regression : {"neg_mean_absolute_error",
        "neg_mean_squared_error", "neg_mean_squared_log_error", "neg_median_absolute_error","r2"}

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


    def evaluate(self, params, df):

        """Evaluates the data.


        Evaluates the data with a given scoring function and given hyper-parameters
        of the whole pipeline. If no parameters are set, default configuration for
        each step is evaluated : no feature selection is applied and no meta features are
        created.

        Parameters
        ----------
        params : dict, default = None.
            Hyper-parameters dictionary for the whole pipeline.

            - The keys must respect the following syntax : "enc__param".

                - "enc" = "ne" for na encoder
                - "enc" = "ce" for categorical encoder
                - "enc" = "fs" for feature selector [OPTIONAL]
                - "enc" = "stck"+str(i) to add layer n°i of meta-features [OPTIONAL]
                - "enc" = "est" for the final estimator

                - "param" : a correct associated parameter for each step. Ex: "max_depth" for "enc"="est", ...

            - The values are those of the parameters. Ex: 4 for key = "est__max_depth", ...

        df : dict, default = None
            Dataset dictionary. Must contain keys and values:

            - "train": pandas DataFrame for the train set.
            - "target" : encoded pandas Serie for the target on train set (with dtype='float' for a regression or dtype='int' for a classification). Indexes should match the train set.

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
        ...     "ne__numerical_strategy" : 0,
        ...     "ce__strategy" : "label_encoding",
        ...     "fs__threshold" : 0.1,
        ...     "stck__base_estimators" : [Regressor(strategy="RandomForest"), Regressor(strategy="ExtraTrees")],
        ...     "est__strategy" : "Linear"
        ... }
        >>> df = {"train" : pd.DataFrame(dataset.data), "target" : pd.Series(dataset.target)}
        >>> opt.evaluate(params, df)
        """

        ne = NA_encoder()
        ce = Categorical_encoder()

        ##########################################
        #    Automatically checking the task
        ##########################################

        # TODO: a lot of code can be factorized for the different tasks

        ##########################################
        #             Classification
        ##########################################

        if (df['target'].dtype == 'int'):

            # Cross validation

            counts = df['target'].value_counts()
            classes_to_drop = counts[counts < self.n_folds].index
            mask_to_drop = df['target'].apply(lambda x: x in classes_to_drop)
            indexes_to_drop = df['target'][mask_to_drop].index
            n_classes = len(counts) - len(classes_to_drop)

            if n_classes == 1:
                raise ValueError("Your target has not enough classes. You can't run the optimiser")

            cv = StratifiedKFold(n_splits=self.n_folds,
                                 shuffle=True,
                                 random_state=self.random_state)

            # Estimator

            est = Classifier()

            # Feature selection if specified

            fs = None
            if (params is not None):
                for p in params.keys():
                    if (p.startswith("fs__")):
                        fs = Clf_feature_selector()
                    else:
                        pass

            # Stacking if specified

            STCK = {}
            if (params is not None):
                for p in params.keys():
                    if (p.startswith("stck")):
                        # TODO: Check if p.split("__")[1] instead?
                        STCK[p.split("__")[0]] = StackingClassifier(verbose=False)  # noqa
                    else:
                        pass

            # Default scoring for classification

            if (self.scoring is None):
                scoring = 'neg_log_loss'  # works also for multiclass pb
                scoring_func = 'neg_log_loss'

            else:
                if (type(self.scoring) == str):
                    if (self.scoring in ["accuracy", "roc_auc", "f1",
                                         "neg_log_loss", "precision", "recall"]):

                        scoring = self.scoring

                        # binary classification
                        if n_classes <= 2:
                            scoring_func = self.scoring

                        # multiclass classification
                        else:
                            pass
                            # TODO !!!

                    else:
                        warnings.warn("Invalid scoring metric. "
                                      "neg_log_loss is used instead.")

                        scoring = 'neg_log_loss'
                        scoring_func = 'neg_log_loss'

                else:
                    scoring = "custom_scoring"
                    scoring_func = self.scoring

        ##########################################
        #               Regression
        ##########################################

        elif (df['target'].dtype == 'float'):

            # Cross validation

            indexes_to_drop = []
            cv = KFold(n_splits=self.n_folds,
                       shuffle=True,
                       random_state=self.random_state)

            # Estimator

            est = Regressor()

            # Feature selection if specified

            fs = None
            if (params is not None):
                for p in params.keys():
                    if (p.startswith("fs__")):
                        fs = Reg_feature_selector()
                    else:
                        pass

            # Stacking if specified

            STCK = {}
            if (params is not None):
                for p in params.keys():
                    if (p.startswith("stck")):
                        # TODO: Check if p.split("__")[1] instead?
                        STCK[p.split("__")[0]] = StackingRegressor(verbose=False)
                    else:
                        pass

            # Default scoring for regression

            if (self.scoring is None):
                scoring = "neg_mean_squared_error"
                scoring_func = "neg_mean_squared_error"

            else:
                if (type(self.scoring) == str):
                    if (self.scoring in ["neg_mean_absolute_error",
                                         "neg_mean_squared_error",
                                         "neg_mean_squared_log_error",
                                         "neg_median_absolute_error",
                                         "r2"]):

                        scoring = self.scoring
                        scoring_func = self.scoring

                    else:
                        warnings.warn("Invalid scoring metric. "
                                      "neg_mean_squarred_error is used instead.")

                        scoring = 'neg_mean_squared_error'
                        scoring_func = 'neg_mean_squared_error'
                else:
                    scoring = "custom_scoring"
                    scoring_func = self.scoring

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
            if ("ce__strategy" in params):
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

        if cache:
            pp = Pipeline(pipe, memory=self.to_path)
        else:
            pp = Pipeline(pipe)

        ##########################################
        #          Fitting the Pipeline
        ##########################################

        start_time = time.time()

        # No params : default configuration

        if (params is None):
            set_params = True
            print('No parameters set. Default configuration is tested')

        else:
            try:
                pp = pp.set_params(**params)
                set_params = True
            except:
                set_params = False

        if (set_params):

            if (self.verbose):
                print("")
                print("#####################################################"
                      " testing hyper-parameters... "
                      "#####################################################")
                print("")
                print(">>> NA ENCODER :" + str(ne.get_params()))
                print("")
                print(">>> CA ENCODER :" + str({'strategy': ce.strategy}))

                if (fs is not None):
                    print("")
                    print(">>> FEATURE SELECTOR :" + str(fs.get_params()))

                for i, stck in enumerate(np.sort(list(STCK))):

                    stck_params = STCK[stck].get_params().copy()
                    stck_params_display = {k: stck_params[k]
                                           for k in stck_params.keys() if
                                           k not in ["level_estimator",
                                                     "verbose",
                                                     "base_estimators"]}

                    print("")
                    print(">>> STACKING LAYER n°"
                          + str(i + 1) + " :" + str(stck_params_display))

                    for j, model in enumerate(stck_params["base_estimators"]):
                        print("")
                        print("    > base_estimator n°" + str(j + 1) + " :"
                              + str(dict(list(model.get_params().items())
                                         + list(model.get_estimator().get_params().items()))))

                print("")
                print(">>> ESTIMATOR :" + str(
                    dict(list(est.get_params().items())
                         + list(est.get_estimator().get_params().items()))
                ))
                print("")

            try:

                # Computing the mean cross validation score across the folds
                scores = cross_val_score(estimator=pp,
                                         X=df['train'].drop(indexes_to_drop),
                                         y=df['target'].drop(indexes_to_drop),
                                         scoring=scoring_func,
                                         cv=cv)
                score = np.mean(scores)

            except:

                scores = [-np.inf for _ in range(self.n_folds)]
                score = -np.inf

        else:
            raise ValueError("Pipeline cannot be set with these parameters."
                             " Check the name of your stages.")

        if (score == -np.inf):
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
            print("MEAN SCORE : " + str(scoring) + " = " + str(score))
            print("VARIANCE : " + str(np.std(scores))
                  + out + "fold " + str(i + 2) + " = " + str(scores[-1]) + ")")
            print("CPU time: %s seconds" % (time.time() - start_time))
            print("")

        return score

    def optimise(self, space, df, max_evals=40):

        """Optimises the Pipeline.

        Optimises hyper-parameters of the whole Pipeline with a given scoring
        function. Algorithm used to optimize : Tree Parzen Estimator.

        IMPORTANT : Try to avoid dependent parameters and to set one feature
        selection strategy and one estimator strategy at a time.

        Parameters
        ----------
        space : dict, default = None.
            Hyper-parameters space:

            - The keys must respect the following syntax : "enc__param".

                - "enc" = "ne" for na encoder
                - "enc" = "ce" for categorical encoder
                - "enc" = "fs" for feature selector [OPTIONAL]
                - "enc" = "stck"+str(i) to add layer n°i of meta-features [OPTIONAL]
                - "enc" = "est" for the final estimator

                - "param" : a correct associated parameter for each step. Ex: "max_depth" for "enc"="est", ...

            - The values must respect the syntax: {"search":strategy,"space":list}

                - "strategy" = "choice" or "uniform". Default = "choice"
                - list : a list of values to be tested if strategy="choice". Else, list = [value_min, value_max].

        df : dict, default = None
            Dataset dictionary. Must contain keys and values:

            - "train": pandas DataFrame for the train set.
            - "target" : encoded pandas Serie for the target on train set (with dtype='float' for a regression or dtype='int' for a classification). Indexes should match the train set.

        max_evals : int, default = 40.
            Number of iterations.
            For an accurate optimal hyper-parameter, max_evals = 40.

        Returns
        -------
        dict.
            The optimal hyper-parameter dictionary.

        Examples
        --------
        >>> from mlbox.optimisation import *
        >>> from sklearn.datasets import load_boston
        >>> #loading data
        >>> dataset = load_boston()
        >>> #optimising the pipeline
        >>> opt = Optimiser()
        >>> space = {
        ...     'fs__strategy':{"search":"choice","space":["variance","rf_feature_importance"]},
        ...     'est__colsample_bytree':{"search":"uniform", "space":[0.3,0.7]}
        ... }
        >>> df = {"train" : pd.DataFrame(dataset.data), "target" : pd.Series(dataset.target)}
        >>> best = opt.optimise(space, df, 3)
        """

        hyperopt_objective = lambda params: -self.evaluate(params, df)

        # Creating a correct space for hyperopt

        if (space is None):
            warnings.warn(
                "Space is empty. Please define a search space. "
                "Otherwise, call the method 'evaluate' for custom settings")
            return dict()

        else:

            if (len(space) == 0):
                warnings.warn(
                    "Space is empty. Please define a search space. "
                    "Otherwise, call the method 'evaluate' for custom settings")
                return dict()

            else:

                hyper_space = {}

                for p in space.keys():

                    if ("space" not in space[p]):
                        raise ValueError("You must give a space list ie values"
                                         " for hyper parameter " + p + ".")

                    else:

                        if ("search" in space[p]):

                            if (space[p]["search"] == "uniform"):
                                hyper_space[p] = hp.uniform(p, np.sort(space[p]["space"])[0],  # noqa
                                                            np.sort(space[p]["space"])[-1])  # noqa

                            elif (space[p]["search"] == "choice"):
                                hyper_space[p] = hp.choice(p, space[p]["space"])
                            else:
                                raise ValueError(
                                    "Invalid search strategy "
                                    "for hyper parameter " + p + ". Please"
                                    " choose between 'choice' and 'uniform'.")

                        else:
                            hyper_space[p] = hp.choice(p, space[p]["space"])

                best_params = fmin(hyperopt_objective,
                                   space=hyper_space,
                                   algo=tpe.suggest,
                                   max_evals=max_evals)

                # Displaying best_params

                for p, v in best_params.items():
                    if ("search" in space[p]):
                        if (space[p]["search"] == "choice"):
                            best_params[p] = space[p]["space"][v]
                        else:
                            pass
                    else:
                        best_params[p] = space[p]["space"][v]

                if (self.verbose):
                    print("")
                    print("")
                    print("~" * 137)
                    print("~" * 57 + " BEST HYPER-PARAMETERS " + "~" * 57)
                    print("~" * 137)
                    print("")
                    print(best_params)

                return best_params
