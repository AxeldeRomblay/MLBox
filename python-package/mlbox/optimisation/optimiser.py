
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import numpy as np
import pandas as pd
import warnings
import time

from hyperopt import fmin, hp, tpe, space_eval
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import *

from ..encoding.na_encoder import *
from ..encoding.categorical_encoder import *
from ..model.supervised.classification.feature_selector import *
from ..model.supervised.regression.feature_selector import *
from ..model.supervised.classification.stacking_classifier import *
from ..model.supervised.regression.stacking_regressor import *
from ..model.supervised.classification.classifier import *
from ..model.supervised.regression.regressor import *


class Optimiser():

    """
    Optimises hyper-parameters of the whole Pipeline :

    1/ NA encoder (missing values encoder)
    2/ CA encoder (categorical features encoder)
    3/ Feature selector [OPTIONAL]
    4/ Stacking estimator - feature engineer [OPTIONAL]
    5/ Estimator (classifier or regressor)

    Works for both regression and classification (multiclass or binary) tasks.


    Parameters
    ----------

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or a scorer callable object / function with signature``scorer(estimator, X, y)``.

        If None, "log_loss" is used for classification and "mean_squarred_error" for regression
	Available scorings for classification : "accuracy","roc_auc", "f1", "log_loss", "precision", "recall"
	Available scorings for regression : "mean_absolute_error", "mean_squared_error", "median_absolute_error", "r2"

    n_folds : int, defaut = 2
        The number of folds for cross validation (stratified for classification)

    random_state : int, defaut = 1
        pseudo-random number generator state used for shuffling

    to_path : str, defaut = "save"
        Name of the folder where models are saved

    verbose : bool, defaut = True
        Verbose mode

    """

    def __init__(self, scoring = None, n_folds = 2, random_state = 1, to_path = "save", verbose = True):

        self.scoring = scoring
        self.n_folds = n_folds
        self.random_state = random_state
        self.to_path = to_path
        self.verbose = verbose


    def get_params(self, deep=True):

        return {'scoring' : self.scoring,
                'n_folds' : self.n_folds,
               'random_state' : self.random_state,
               'verbose' : self.verbose}


    def set_params(self,**params):

        self.__fitOK = False

        for k,v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for optimiser Optimiser. Parameter IGNORED. Check the list of available parameters with `optimiser.get_params().keys()`")
            else:
                setattr(self,k,v)


    def evaluate(self, params, df):

        '''

        Evaluates the scoring function with given hyper-parameters of the whole Pipeline. If no parameters are set, defaut configuration for each step is evaluated : no feature selection is applied and no meta features are created.


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
            Train dictionnary. Must contain keys "train" and "target" with the train dataset (pandas DataFrame) and the associated target (pandas Serie with
            dtype='float' for a regression or dtype='int' for a classification) resp.


        Returns
        -------

        score : float.
            The score. The higher the better (positive for a score and negative for a loss)


        '''


        ne = NA_encoder()
        ce = Categorical_encoder()


        ##########################################################################
        #################### automatically checking the task ####################
        ##########################################################################


        ######################
        ### classification ###

        if(df['target'].dtype=='int'):

            ### cross validation ###

            counts = df['target'].value_counts()
            classes_to_drop = counts[counts<self.n_folds].index
            indexes_to_drop = df['target'][df['target'].apply(lambda x: x in classes_to_drop)].index

            cv = StratifiedKFold(n_splits = self.n_folds, shuffle=True,random_state=self.random_state)

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
                        STCK[p.split("__")[0]] = StackingClassifier(verbose=False)
                    else:
                        pass


            ### defaut scoring for classification ###

            auc = False

            if(self.scoring is None):
                self.scoring = 'log_loss'

            elif(self.scoring=='roc_auc'):
                auc = True
                self.scoring = make_scorer(lambda y_true, y_pred: roc_auc_score(pd.get_dummies(y_true), y_pred), greater_is_better=True, needs_proba=True)

            else:
                if(type(self.scoring)==str):
                    if(self.scoring in ["accuracy","roc_auc", "f1", "log_loss", "precision", "recall"]):
                        pass
                    else:
                        warnings.warn("Invalid scoring metric. log_loss is used instead.")
                        self.scoring = 'log_loss'

                else:
                    pass


        ##################
        ### regression ###

        elif(df['target'].dtype=='float'):

            ### cross validation ###

            indexes_to_drop = []
            cv = KFold(n_splits = self.n_folds, shuffle=True,random_state=self.random_state)

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
                        STCK[p.split("__")[0]] = StackingRegressor(verbose=False)
                    else:
                        pass

            ### defaut scoring for regression ###

            auc = False

            if(self.scoring is None):
                self.scoring = "mean_squared_error"
            else:
                if(type(self.scoring)==str):
                    if(self.scoring in ["mean_absolute_error", "mean_squared_error", "median_absolute_error", "r2"]):
                        pass
                    else:
                        warnings.warn("Invalid scoring metric. mean_squarred_error is used instead.")
                        self.scoring = 'mean_squared_error'
                else:
                    pass

        else:
            raise ValueError("Impossible to determine the task. Please check that your target is encoded.")


        ############################################################
        ##################### creating the pipeline ##################
        ############################################################

        pipe = [("ne",ne),("ce",ce)]

        if(fs is not None):
            pipe.append(("fs",fs))
        else:
            pass

        for stck in np.sort(STCK.keys()):
            pipe.append((stck,STCK[stck]))

        pipe.append(("est",est))
        pp = Pipeline(pipe)


        ############################################################
        #################### fitting the pipeline ###################
        ############################################################

        start_time = time.time()

        ### no params : defaut config ###
        if(params is None):
            set_params = True
            print('No parameters set. Default configuration is tested')

        else:
            try:
                pp = pp.set_params(**params)
                set_params = True
            except:
                set_params = False

        if(set_params):


            if(self.verbose):
                print("")
                print("########################################################## testing hyper-parameters... #################################################################")
                print("")
                print(">>> NA ENCODER :"+str(ne.get_params()))
                print("")
                print(">>> CA ENCODER :"+str({'strategy': ce.strategy}))

                if(fs is not None):
                    print("")
                    print(">>> FEATURE SELECTOR :"+str(fs.get_params()))

                for i, stck in enumerate(np.sort(STCK.keys())):

                    stck_params = STCK[stck].get_params().copy()
                    stck_params_display = {k:stck_params[k] for k in stck_params.keys() if k not in ["level_estimator", "verbose", "base_estimators"]}

                    print("")
                    print(">>> STACKING LAYER n°"+str(i+1)+" :"+str(stck_params_display))
                    for j, model in enumerate(stck_params["base_estimators"]):
                        print("")
                        print("    > base_estimator n°"+str(j+1)+" :"+str(dict(model.get_params().items()+model.get_estimator().get_params().items())))

                print("")
                print(">>> ESTIMATOR :"+str(dict(est.get_params().items()+est.get_estimator().get_params().items())))
                print("")

            try:

                ### computing the mean cross validation score across the folds ###
                scores = cross_val_score(estimator = pp, X = df['train'].drop(indexes_to_drop), y = df['target'].drop(indexes_to_drop), scoring = self.scoring, cv = cv)
                score = np.mean(scores)

            except:
                scores = [-np.inf for fold in range(self.n_folds)]
                score = -np.inf
        else:
            raise ValueError("Pipeline cannot be set with these parameters. Check the name of your stages.")

        if(score==-np.inf):
            warnings.warn("An error occurred while computing the cross validation mean score. Check the parameter values and your scoring function.")


        ############################################################
        ##################### reporting score ####################
        ############################################################

        out = " ("

        for i, s in enumerate(scores[:-1]):
            out = out+"fold "+str(i+1)+" = "+str(s)+", "

        if(auc):
            self.scoring = "roc_auc"

        if(self.verbose):
            print("")
            print("MEAN SCORE : "+str(self.scoring)+" = "+str(score))
            print("VARIANCE : "+str(np.std(scores))+out+"fold "+str(i+2)+" = "+str(scores[-1])+")")
            print("CPU time: %s seconds" % (time.time() - start_time))
            print("")


        return score




    def optimise(self, space, df, max_evals = 40):

        '''

        Optimises hyper-parameters of the whole Pipeline with a given scoring function. By defaut, estimator used is 'xgboost' and no feature selection is applied.
        Algorithm used to optimize : Tree Parzen Estimator (http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html)
        IMPORTANT : Try to avoid dependent parameters and to set one feature selection strategy and one estimator strategy at a time.

        Parameters
        ----------

        space : dict, defaut = None.
            Hyper-parameters space.

            - The keys must respect the following syntax : "enc__param".

            With :
                1/ "enc" = "ne" for na encoder
                2/ "enc" = "ce" for categorical encoder
                3/ "enc" = "fs" for feature selector [OPTIONAL]
                4/ "enc" = "stck"+str(i) to add layer n°i of meta-features (assuming 1 ... i-1 layers are created...) [OPTIONAL]
                5/ "enc" = "est" for the final estimator

            And:
                "param" : a correct associated parameter for each step. (for example : "max_depth" for "enc"="est", "entity_embedding" for "enc"="ce")

            - The values must respect the following syntax : {"search" : strategy, "space" : list}

            With:
                "strategy" = "choice" or "uniform". Defaut = "choice"

            And:
               list : a list of values to be tested if strategy="choice". If strategy = "uniform", list = [value_min, value_max].


        df : dict, defaut = None
            Train dictionnary. Must contain keys "train" and "target" with the train dataset (pandas DataFrame) and the associated target (pandas Serie with
            dtype='float' for a regression or dtype='int' for a classification) resp.


        max_evals : int, defaut = 40.
            Number of iterations. For an accurate optimal hyper-parameter, max_evals = 40.


        Returns
        -------

        best_params : dict.
            The optimal hyper-parameter dictionnary.


        '''


        hyperopt_objective = lambda params: -self.evaluate(params, df)


        ### creating a correct space for hyperopt ###

        if(space is None):
            warnings.warn("Space is empty. Please define a search space. Otherwise, call the method 'evaluate' for custom settings")
            return dict()

        else:

            if(len(space)==0):
                warnings.warn("Space is empty. Please define a search space. Otherwise, call the method 'evaluate' for custom settings")
                return dict()

            else:

                hyper_space = {}

                for p in space.keys():

                    if("space" not in space[p]):
                        raise ValueError("You must give a space list ie values for hyper parameter "+p+".")

                    else:

                        if("search" in space[p]):

                            if(space[p]["search"]=="uniform"):
                                hyper_space[p] = hp.uniform(p,np.sort(space[p]["space"])[0],np.sort(space[p]["space"])[-1])

                            elif(space[p]["search"]=="choice"):
                                hyper_space[p] = hp.choice(p,space[p]["space"])
                            else:
                                raise ValueError("Invalid search strategy for hyper parameter "+p+". Please choose between 'choice' and 'uniform'.")

                        else:
                            hyper_space[p] = hp.choice(p,space[p]["space"])


                best_params = fmin(hyperopt_objective, space=hyper_space, algo=tpe.suggest, max_evals=max_evals)


                ### displaying best_params ###

                for p,v in best_params.iteritems():
                    if("search" in space[p]):
                        if(space[p]["search"]=="choice"):
                            best_params[p] = space[p]["space"][v]
                        else:
                            pass
                    else:
                        best_params[p] = space[p]["space"][v]

                if(self.verbose):
                    print("")
                    print("")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BEST HYPER-PARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    print("")
                    print(best_params)


                return best_params

