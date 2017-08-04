# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import os
import time
from sklearn.pipeline import Pipeline
from .drift import DriftThreshold
from ..encoding.na_encoder import NA_encoder
from ..encoding.categorical_encoder import Categorical_encoder


class Drift_thresholder():

    """Automatically drops ids and drifting variables between train and test datasets.

    Drops on train and test datasets. The list of drift coefficients is available and
    saved as "drifts.txt". To get familiar with drift:
    https://github.com/AxeldeRomblay/MLBox/blob/master/docs/webinars/slides.pdf

    Parameters
    ----------
    threshold : float, defaut = 0.6
        Drift threshold under which features are kept. Must be between 0. and 1.
        The lower the more you keep non-drifting/stable variables: a feature with
        a drift measure of 0. is very stable and a one with 1. is highly unstable.

    inplace : bool, default = False
        If True, train and test datasets are transformed. Returns self.
        Otherwise, train and test datasets are not transformed. Returns a new dictionnary with
        cleaned datasets.

    verbose : bool, default = True
        Verbose mode

    to_path : str, default = "save"
        Name of the folder where the list of drift coefficients is saved.
    """

    def __init__(self,
                 threshold=0.6,
                 inplace=False,
                 verbose=True,
                 to_path="save"):

        self.threshold = threshold
        self.inplace = inplace
        self.verbose = verbose
        self.to_path = to_path
        self.__Ddrifts = {}
        self.__fitOK = False


    def fit_transform(self, df):

        """Fits and transforms train and test datasets

        Automatically drops ids and drifting variables between train and test datasets.
        The list of drift coefficients is available and saved as "drifts.txt"

        Parameters
        ----------
        df : dict, defaut = None
            Dictionnary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set

        Returns
        -------
        dict
            Dictionnary containing :

            - 'train' : transformed pandas dataframe for train dataset
            - 'test' : transformed pandas dataframe for test dataset
            - 'target' : pandas serie for the target on train set
        """

        ######################################################
        #                   Deleting IDs
        ######################################################

        # Exception

        if (df["test"].shape[0] == 0):
            if (self.verbose):
                print("")
                print("You have no test dataset...")

            return df

        else:

            start_time = time.time()

            ds = DriftThreshold(self.threshold)
            na = NA_encoder(numerical_strategy=0)
            ca = Categorical_encoder()

            pp = Pipeline([("na", na), ("ca", ca)])
            pp.fit(df['train'], None)

            # Deleting IDs with drift threshold method

            if (self.verbose):
                print("")
                print("computing drifts...")

            ds.fit(pp.transform(df['train']), pp.transform(df['test']))

            if (self.verbose):
                print("CPU time: %s seconds" % (time.time() - start_time))
                print("")

            self.__fitOK = True
            self.__Ddrifts = ds.drifts()
            drifts_top = sorted(ds.drifts().items(),
                                key=lambda x: x[1],
                                reverse=True)[:10]

            if (self.verbose):
                print("Top 10 drifts")
                print("")
                for d in range(len(drifts_top)):
                    print(drifts_top[d])

            if (self.verbose):
                print("")
                print("deleted "
                      "variables : " + str(ds.get_support(complement=True)))

            ######################################################
            #           Dumping Encoders into directory
            ######################################################

            if (self.to_path is not None):

                try:
                    os.mkdir(self.to_path)
                except OSError:
                    pass

                if (self.verbose):
                    print("")
                    print("dumping drift coefficients into "
                          "directory : " + self.to_path)

                file = open(self.to_path + '/drifts.txt', "w")
                file.write("\n")
                file.write(
                    "*****************************************************"
                    "  DRIFTS Coefficients "
                    "*****************************************************\n")
                file.write("\n")

                for var, d in sorted(ds.drifts().items(),
                                     key=lambda x: x[1],
                                     reverse=True):
                    file.write(str(var) + " = " + str(d) + '\n')

                file.close()

                if (self.verbose):
                    print("drift coefficients dumped")

            # Returning datasets with no IDs

            if (self.inplace):

                df['train'] = ds.transform(df['train'])
                df['test'] = ds.transform(df['test'])

            else:

                return {'train': ds.transform(df['train']),
                        'test': ds.transform(df['test']),
                        'target': df['target']}

    def drifts(self):

        """Returns the univariate drifts for all variables.

        Returns
        -------
        dict
            Dictionnary containing the drifts for each feature
        """

        if self.__fitOK:

            return self.__Ddrifts
        else:
            raise ValueError('Call the fit_transform function before !')
