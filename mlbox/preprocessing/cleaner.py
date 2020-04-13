# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import sys
import os
import warnings
import time

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from .drift import DriftThreshold
from ..optimisation.encoding.na_encoder import NanEncoder
from ..optimisation.encoding.categorical_encoder import CategoricalEncoder


class Cleaner():

    """Cleans data :

    - drops specific features
    - drops duplicates
    - drops drifting features
    - drops sparse features/samples
    - drops constant features
    - drops correlations

    Parameters
    ----------
    to_path : str, default = "save"
        Name of the folder where computations are saved.

    verbose : bool, default = True
        Verbose mode
    """

    def __init__(self,
                 to_path="save",
                 verbose=True):

        self.to_path = to_path
        self.verbose = verbose


    def drop_feature(self, data, L=[]):

        """Drops specific features on training and test sets.

        Parameters
        ----------
        data : dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set

        L : list, default = []
            List of features to drop

        Returns
        -------
        dict
            Dictionary containing :

            - 'train' : transformed pandas dataframe for train dataset
            - 'test' : transformed pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set
        """

        if (len(L) == 0):

            warnings.warn("List is empty. No specific features will be dropped")

            return data

        else:

            start_time = time.time()

            if (self.verbose):
                sys.stdout.write("dropping specific features ...")

            # checking L

            to_drop = list(set(data["train"].columns) &
                           set(data["test"].columns) &
                           set(L))

            if(len(to_drop) < len(L)):
                warnings.warn("Some features to be dropped do not exist. Please check the list !")

            if (self.verbose):
                sys.stdout.write(" - " + str(np.round(time.time() - start_time, 2)) + " s")
                print("")

            # display

            if (self.verbose):
                print("")
                print("> Number of features dropped : " + str(len(to_drop)))
                print("")

            return {"train" : data["train"].drop(to_drop, axis=1),
                    "test" : data["test"].drop(to_drop, axis=1),
                    "target" : data["target"]
                    }


    def drop_duplicate(self, data, drop=True):

        """Drops training duplicates.

        Parameters
        ----------
        data : dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set

        drop : bool, default = True
            If True, drops training duplicates

        Returns
        -------
        dict
            Dictionary containing :

            - 'train' : transformed pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set
        """

        if (drop==False):

            warnings.warn("No training duplicates will be dropped")

            return data

        else:

            start_time = time.time()

            if (self.verbose):
                sys.stdout.write("dropping duplicates on the training set ...")

            n_samples = data["train"].shape[0]
            i = data["train"].duplicated()

            if (self.verbose):
                sys.stdout.write(" - " +
                                 str(np.round(time.time() - start_time, 2)) + " s")
                print("")

            # display

            if (self.verbose):
                print("")
                print("> Number of training samples dropped : " +
                      str(i[i == True].shape[0]) + " out of " + str(n_samples))
                print("")

            return {"train": data["train"].loc[i[i == False].index],
                    "test": data["test"],
                    "target": data["target"]
                    }


    def drop_sparsity(self, data, threshold=0.95):

        """Drops sparse training samples and sparse features.

        Parameters
        ----------
        data : dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set

        threshold : float, default = 0.95
            Threshold (%). Samples and features with higher sparsity than the threshold are dropped.


        Returns
        -------
        dict
            Dictionary containing :

            - 'train' : transformed pandas dataframe for train dataset
            - 'test' : transformed pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set
        """

        if (threshold >= 1):

            warnings.warn("Sparsity threshold is higher than 1. No samples and no features will be dropped")

            return data

        else:

            start_time = time.time()

            if (self.verbose):
                sys.stdout.write("dropping sparse features and sparse samples ...")

            n_samples = data["train"].shape[0]
            n_features = data["train"].shape[1]

            # drops samples
            i = data["train"].dropna(axis=0,
                                     thresh=n_features*(1-threshold)).index

            # drops features
            f = (data["train"].isnull().sum() * 1. / data["train"].shape[0])

            if (self.verbose):
                sys.stdout.write(" - " +
                                 str(np.round(time.time() - start_time, 2)) + " s")
                print("")

            # save

            if (self.to_path is not None):

                try:
                    os.mkdir(self.to_path)
                except OSError:
                    pass

                file = open(self.to_path + '/na.txt', "w")
                file.write("\n")
                file.write(
                    "*******************************************"
                    "  SPARSE features on train (%)"
                    "*******************************************\n")
                file.write("\n")
                for col, s in sorted(dict(f).items(),
                                     key=lambda x: x[1],
                                     reverse=True):
                    file.write(str(col) + " = " + str(s) + '\n')
                file.close()

            # display

            if (self.verbose):
                print("")
                print("> Number of training samples dropped : " +
                      str(n_samples-len(i)) + " out of " + str(n_samples))
                print("> Number of features dropped : "
                      + str(f[f>threshold].shape[0]))
                print("> Top 10 sparse features (%) : ")
                print(f.sort_values(ascending=False)[:10])
                print("")

            return {"train" : data["train"].loc[i][f[f<=threshold].index],
                    "test" : data["test"][f[f<=threshold].index],
                    "target" : data["target"]
                    }


    def drop_constant(self, data, drop=True):

        """Drops constant features.

        Parameters
        ----------
        data : dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set

        drop : bool, default = True
            If True, drops constant features

        Returns
        -------
        dict
            Dictionary containing :

            - 'train' : transformed pandas dataframe for train dataset
            - 'test' : transformed pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set
        """

        if (drop == False):

            warnings.warn("No constant features will be dropped")

            return data

        else:

            start_time = time.time()

            if (self.verbose):
                sys.stdout.write("dropping constant features ...")

            Lconstants = []

            for col in data["train"].columns:
                if (data["train"][col].nunique(dropna=False) == 1):
                    Lconstants.append(col)

            if (self.verbose):
                sys.stdout.write(" - " +
                                 str(np.round(time.time() - start_time, 2)) + " s")
                print("")

            # save

            if (self.to_path is not None):

                try:
                    os.mkdir(self.to_path)
                except OSError:
                    pass

                file = open(self.to_path + '/constant.txt', "w")
                file.write("\n")
                file.write(
                    "*******************************************"
                    "  CONSTANT features on train"
                    "*******************************************\n")
                file.write("\n")
                for col in Lconstants:
                    file.write(str(col) + '\n')
                file.close()

            # display

            if (self.verbose):
                print("")
                print("> Number of features dropped : " + str(len(Lconstants)))
                if (len(Lconstants) > 0):
                    print("> Top 10 constant features : ")
                    for col in Lconstants[:10]:
                        print(col)
                print("")

            return {"train" : data["train"].drop(Lconstants, axis=1),
                    "test" : data["test"].drop(Lconstants, axis=1),
                    "target" : data["target"]
                    }


    def drop_drift(self, data, threshold=0.6):

        """Drops drifting features (ids, ...)

        Parameters
        ----------
        data : dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set

        threshold : float, default = 0.6
            Threshold (%). Features with higher drift coefficients than the threshold are dropped.

        Returns
        -------
        dict
            Dictionary containing :

            - 'train' : transformed pandas dataframe for train dataset
            - 'test' : transformed pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set
        """

        if (threshold >= 1):

            warnings.warn("Drift threshold is higher than 1. No features will be dropped")

            return data

        else:

            start_time = time.time()

            if (self.verbose):
                sys.stdout.write("dropping drifting features ...")

            if (data["test"].shape[0] == 0):

                if (self.verbose):
                    sys.stdout.write(" - " +
                                     str(np.round(time.time() - start_time, 2)) + " s")
                    print("")
                    print("")
                    print("> No features dropped (you have no test set)")
                    print("")

                return data

            else:

                ds = DriftThreshold(threshold)
                na = NanEncoder(numerical_strategy=0)
                ca = CategoricalEncoder()

                pp = Pipeline([("na", na), ("ca", ca)])
                pp.fit(data['train'], None)

                ds.fit(pp.transform(data['train']), pp.transform(data['test']))

                Ddrifts = ds.drifts()
                drifts_sorted = pd.Series(Ddrifts).sort_values(ascending=False)

                if (self.verbose):
                    sys.stdout.write(" - " +
                                     str(np.round(time.time() - start_time, 2)) + " s")
                    print("")

                # save

                if (self.to_path is not None):

                    try:
                        os.mkdir(self.to_path)
                    except OSError:
                        pass

                    file = open(self.to_path + '/drifts.txt', "w")
                    file.write("\n")
                    file.write(
                        "*******************************************"
                        "  DRIFTS coefficients "
                        "*******************************************\n")
                    file.write("\n")

                    for col, d in sorted(Ddrifts.items(),
                                         key=lambda x: x[1],
                                         reverse=True):
                        file.write(str(col) + " = " + str(d) + '\n')

                    file.close()

                # display

                if (self.verbose):
                    print("")
                    print("> Number of features dropped : " +
                          str(len(ds.get_support(complement=True))))
                    print("> Top 10 drifting features (%) : ")
                    print("")
                    print(drifts_sorted.head(10))
                    print("")

                return {'train': ds.transform(data['train']),
                        'test': ds.transform(data['test']),
                        'target': data['target']}


    def drop_correlation(self, data, threshold=0.95):

        """Drops highly-correlated features.

        Parameters
        ----------
        data : dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set

        threshold : float, defaut = 0.95
            Threshold (%). Features with higher correlation coefficients than the threshold are dropped.

        Returns
        -------
        dict
            Dictionary containing :

            - 'train' : transformed pandas dataframe for train dataset
            - 'test' : transformed pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set
        """

        if (threshold >= 1):

            warnings.warn("Correlation threshold is higher than 1. No features will be dropped")

            return data

        else:

            start_time = time.time()

            if (self.verbose):
                sys.stdout.write("dropping highly-correlated features ...")

            Lcorr = []

            corr = data["train"].corr().abs()
            corr = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))).stack().sort_values(ascending=False)

            for (f1, f2) in dict(corr[corr > threshold]).keys():

                if ((f1 not in Lcorr) & (f2 not in Lcorr)):
                    Lcorr.append(f1)

                else:
                    pass

            if (self.verbose):
                sys.stdout.write(" - " + str(np.round(time.time() - start_time, 2)) + " s")
                print("")

                # save

                if (self.to_path is not None):

                    try:
                        os.mkdir(self.to_path)
                    except OSError:
                        pass

                    file = open(self.to_path + '/correlations.txt', "w")
                    file.write("\n")
                    file.write(
                        "*******************************************"
                        "  CORRELATIONS coefficients "
                        "*******************************************\n")
                    file.write("\n")

                    for cols, c in dict(corr).items():
                        file.write(str(cols) + " = " + str(c) + '\n')

                    file.close()

                # display

                if (self.verbose):
                    print("")
                    print("> Number of features dropped : " + str(len(Lcorr)))
                    print("> Top 10 correlated features (%) : ")
                    print("")
                    print(corr.head(10))
                    print("")

                return {"train": data["train"].drop(Lcorr, axis=1),
                        "test": data["test"].drop(Lcorr, axis=1),
                        "target": data["target"]
                        }

    def run(self, data,
            drop_feature=[],
            drop_duplicate=True,
            sparsity_threshold=0.95,
            drop_constant=True,
            drift_threshold=0.6,
            correlation_threshold=0.95):

        """Runs the following steps :

        - dropping specific features
        - dropping duplicates
        - dropping sparse samples and sparse features
        - dropping constant features
        - dropping drifting features
        - dropping highly-correlated features

        Parameters
        ----------
        data : dict, defaut = None
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set

        drop_feature : list, default = []
            List of features to drop

        drop_duplicate : bool, default = True
            If True, drop duplicates

        sparsity_threshold : float, default = 0.95
            Threshold (%). Features and samples with higher sparsity than the threshold are dropped.

        drop_constant : bool, default = True
            If True, drop constant features

        drift_threshold : float, default = 0.6
            Threshold (%). Features with higher drift coefficients than the threshold are dropped.

        correlation_threshold : float, default = 0.95
            Threshold (%). Features with higher correlation coefficients than the threshold are dropped.

        Returns
        -------
        dict
            Dictionary containing :

            - 'train' : transformed pandas dataframe for train dataset
            - 'test' : transformed pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set
        """

        start_time = time.time()

        if (self.verbose):
            print("")
            print("STEP 4 - cleaning the databases ...")
            print("")
            print("")

        return self.drop_correlation(
                    self.drop_drift(
                        self.drop_constant(
                            self.drop_sparsity(
                                self.drop_duplicate(
                                    self.drop_feature(data,
                                                      drop_feature),
                                    drop_duplicate),
                                sparsity_threshold),
                            drop_constant),
                        drift_threshold),
                    correlation_threshold)
