# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.externals.joblib import Parallel, delayed
from ..optimisation.encoding.target_encoder import TargetEncoder

# TODO : enlever l'encoding de target (et modif cat encoder)
# TODO  : donner la description des features (avec accesseur)

def detect_type(df):

    Lnum = []
    Lcat = []
    Ltext = []
    Lind = []
    Llist = []
    Ldate = []

    for col in df.columns:

        # date
        if (df[col].dtype == 'datetime64[ns]'):

            Ldate.append(col)

        else:

            try:

                # numerical (float/int), bool and numerical index
                serie = df[col].apply(float)
                Lnum.append(col)

            except:

                # object
                try:

                    # date
                    serie = pd.DatetimeIndex(pd.to_datetime(df[col]))
                    Ldate.append(col)

                except:

                    # list
                    if(df[col].apply(lambda x: type(x) == list).sum() > 0):

                        Llist.append(col)

                    else:

                        try:

                            max_token = df[col].apply(lambda x: len(x.split(" "))).max()

                            if (max_token > 2):

                                # text
                                Ltext.append(col)

                            else:

                                max_count = df[col].value_counts()[0]

                                if (max_count == 1):

                                    # categorical index
                                    Lind.append(col)

                                else:

                                    # categorical features
                                    Lcat.append(col)
                        except:

                            # weird features
                            warnings.warn("'" + col + "' is a weird feature ... It will be ignored.")

    return {
        "numerical" : Lnum,
        "categorical" : Lcat,
        "index" : Lind,
        "list" : Llist,
        "text" : Ltext,
        "date" : Ldate
    }


def convert_list(serie):

    """Converts lists in a pandas serie into a dataframe
    where which element of a list is a column

    Parameters
    ----------
    serie : pandas Serie
        The serie you want to cast into a dataframe

    Returns
    -------
    pandas DataFrame
        The converted dataframe
    """

    import numpy
    import pandas

    if (serie.apply(lambda x: type(x) == list).sum() > 0):

        serie = serie.apply(lambda x: [x] if type(x) != list else x)
        cut = int(numpy.percentile(serie.apply(len), 90))  # TODO: To test

        serie = serie.apply(lambda x: x[:cut])

        return pandas.DataFrame(serie.tolist(),
                                index=serie.index,
                                columns=[serie.name + "_item" + str(i + 1)
                                         for i in range(cut)]
                                )

    else:

        return serie


def convert_float_and_dates(serie):

    """Converts into float if possible and converts dates.

    Creates timestamp from 01/01/2017, year, month, day, day_of_week and hour

    Parameters
    ----------
    serie : pandas Serie
        The serie you want to convert

    Returns
    -------
    pandas DataFrame
        The converted dataframe
    """

    import pandas

    # dtype is already a date

    if (serie.dtype == 'datetime64[ns]'):

        df = pandas.DataFrame([], index=serie.index)
        df[serie.name + "_timestamp"] = (pandas.DatetimeIndex(serie) -
                                         pandas.datetime(2017, 1, 1)
                                         ).total_seconds()

        df[serie.name + "_year"] = pandas.DatetimeIndex(serie).year.astype(float)
        # TODO: be careful with nan ! object or float ??

        df[serie.name + "_month"] = pandas.DatetimeIndex(serie).month.astype(float)
        # TODO: be careful with nan ! object or float ??

        df[serie.name + "_day"] = pandas.DatetimeIndex(serie).day.astype(float)
        # TODO: be careful with nan ! object or float ??

        df[serie.name + "_dayofweek"] = pandas.DatetimeIndex(serie).dayofweek.astype(float)
        # TODO: be careful with nan ! object or float ??

        df[serie.name + "_hour"] = pandas.DatetimeIndex(serie).hour.astype(float) + \
                                   pandas.DatetimeIndex(serie).minute.astype(float) / 60. + \
                                   pandas.DatetimeIndex(serie).second.astype(float) / 3600.

        return df

    else:

        # Convert float

        try:
            serie = serie.apply(float)

        except:
            pass

        # Cleaning/converting dates

        if (serie.dtype != 'object'):
            return serie

        else:
            # trying to cast into date
            df = pandas.DataFrame([], index=serie.index)

            try:

                serie_to_df = pandas.DatetimeIndex(pd.to_datetime(serie))

                df[serie.name + "_timestamp"] = (serie_to_df -
                                                 pandas.datetime(2017, 1, 1)
                                                 ).total_seconds()

                df[serie.name + "_year"] = serie_to_df.year.astype(
                    float)  # TODO: be careful with nan ! object or float??

                df[serie.name + "_month"] = serie_to_df.month.astype(float)
                # TODO: be careful with nan ! object or float??

                df[serie.name + "_day"] = serie_to_df.day.astype(float)
                # TODO: be careful with nan ! object or float??

                df[serie.name + "_dayofweek"] = serie_to_df.dayofweek.astype(float)
                # TODO: be careful with nan ! object or float??

                df[serie.name + "_hour"] = serie_to_df.hour.astype(float) + \
                                           serie_to_df.minute.astype(float) / 60. + \
                                           serie_to_df.second.astype(float) / 3600.

                return df

            except:

                return serie


class Reader():

    """Reads, splits, crunches and casts data

    Parameters
    ----------
    dump : bool, default = False
        If True, dumps each file to hdf5 format.

    to_path : str, default = "save"
        Name of the folder where files and encoders are saved.

    verbose : bool, default = True
        Verbose mode
    """

    def __init__(self,
                 dump=False,
                 to_path="save",
                 verbose=True):

        self.dump = dump
        self.to_path = to_path
        self.verbose = verbose
        self.__stats = None

    def get_stats(self):

        return self.__stats

    def __crunch_train(self, df1, df2):

        """Crunches two different training datasets into one. Recursive crunch.

        Parameters
        ----------
        df1 : pandas DataFrame
            The first training dataset on which you crunch the second training dataset

        df2 : pandas DataFrame
            The second training dataset

        Returns
        -------
        pandas DataFrame
            The reduced main training dataset.
        """

        # ANALYSING INDEXES

        has_index1 = False
        has_index2 = False

        if(any([isinstance(x, str) for x in df1.index])):
            has_index1 = True

        if (any([isinstance(x, str) for x in df2.index])):
            has_index2 = True

        # CRUNCHING

        s1 = set(df1.columns)
        s2 = set(df2.columns)
        s = s1 & s2

        ########################################################
        # FIRST CASE : df1 and df2 have exactly the same columns
        ########################################################

        if (s1 == s2):

            if (len(s) < 10):
                warnings.warn("Crunching on few features: " + str(list(s)) + " .The result might not be exact")

            if ((has_index1 == False) & (has_index2 == False)):
                return pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)

            elif ((has_index1 == False) & (has_index2 == True)):
                return pd.concat([df1, df2]).drop_duplicates(keep='last')

            elif ((has_index1 == True) & (has_index2 == False)):
                return pd.concat([df1, df2]).drop_duplicates(keep='first')

            else:
                return pd.concat([df1, df2]).drop_duplicates(keep='last')

        else:

            ################################################################
            # SECOND CASE : df2 columns is included in df1 columns (strictly)
            #################################################################

            if (s == s2):

                if (len(s) < 10):
                    warnings.warn("Crunching on few features: " + str(list(s)) + " .The result might not be exact")

                if ((has_index1 == False) & (has_index2 == False)):
                    return pd.concat([df1, df2]).drop_duplicates(subset=list(s2), keep='first').reset_index(drop=True)

                elif ((has_index1 == True) & (has_index2 == False)):
                    return pd.concat([df1, df2]).drop_duplicates(subset=list(s2), keep='first')

                else:
                    warnings.warn("Crunching might drop some indexes...")
                    return pd.concat([df1, df2]).drop_duplicates(subset=list(s2), keep='first')

            ###############################################################
            # THIRD CASE: df1 columns is included in df2 columns (strictly)
            ###############################################################

            elif (s == s1):

                if (len(s) < 10):
                    warnings.warn("Crunching on few features: " + str(list(s)) + " .The result might not be exact")

                if ((has_index1 == False) & (has_index2 == False)):
                    return pd.concat([df1, df2]).drop_duplicates(subset=list(s1), keep='last').reset_index(drop=True)

                elif ((has_index1 == False) & (has_index2 == True)):
                    return pd.concat([df1, df2]).drop_duplicates(subset=list(s1), keep='last')

                else:
                    warnings.warn("Crunching might drop some indexes...")
                    return pd.concat([df1, df2]).drop_duplicates(subset=list(s1), keep='last')

            ######################################################
            # LAST CASE : df1 columns and df2 columns are disjoints
            ######################################################

            else:

                # no common features
                if (len(s) == 0):

                    if (has_index1 & has_index2):
                        warnings.warn("Crunching on separate sets of features ! Merging on indexes...")

                        return pd.merge(df1, df2,
                                        how='outer',
                                        left_index=True,
                                        right_index=True)

                    else:

                        if(df1.shape[0] == df2.shape[0]):
                            warnings.warn("Crunching on separate sets of features ! The result may not be exact...")

                            return pd.concat([df1, df2], axis=1)

                        else:

                            if ((has_index1 == False) & (has_index2 == False)):
                                return pd.concat([df1, df2]).reset_index(drop=True)

                            else:
                                return pd.concat([df1, df2])

                #there is a common subset of features
                else:

                    # at least one dataset has no indexes
                    if ((has_index1 == False) | (has_index2 == False)):

                        if ((has_index1 == False) & (has_index2 == False)):
                            df2.index = range(df1.shape[0], df1.shape[0]+df2.shape[0])

                        duplicates = pd.concat([df1, df2]).duplicated(subset=list(s), keep=False)
                        i_duplicates = duplicates[duplicates == True].index

                        i_squash1 = list(set(i_duplicates) & set(df1.index))
                        i_squash2 = list(set(i_duplicates) & set(df2.index))

                        #duplicates come from one dataset only
                        if ((len(i_squash1)==0) | (len(i_squash2)==0)):

                            return pd.concat([df1, df2])

                        else:

                            if (len(s) < 10):
                                warnings.warn("Crunching on few features: " + str(list(s)) + " .The result might not be exact")

                            if ((has_index1 == True) | (has_index2 == True)):
                                warnings.warn("Crunching might drop some indexes...")

                            return pd.concat([df1.drop(i_squash1),
                                              df2.drop(i_squash2),
                                              pd.merge(df1.loc[i_squash1],
                                                       df2.loc[i_squash2],
                                                       on = list(s),
                                                       how = "outer")
                                              ])

                    #both datasets have indexes
                    else:
                        indexes1 = [i for i in df1.index if(type(i)==str)]

                        #df1 only contains indexes (df2 too since it is recursive...)
                        if (df1.drop(indexes1).shape[0] == 0):

                            return pd.merge(df1, df2, left_index=True, right_index=True, on = list(s), how="outer")

                        # general case : df1 contains some indexes and some int, df2 contains (only) indexes.
                        else:

                            return pd.concat([pd.merge(df1.loc[indexes1],
                                                       df2,
                                                       left_index=True,
                                                       right_index=True,
                                                       on=list(s),
                                                       how="left"),
                                              self.__crunch_train(df1.drop(indexes1),
                                                                  df2.drop(list(set(indexes1) & set(df2.index)))
                                                                  )
                                              ])

    def __crunch_test(self, df1, df2):

        """Crunches two different test datasets into one. Recursive crunch.

        Parameters
        ----------
        df1 : pandas DataFrame
            The first test dataset on which you crunch the second test dataset

        df2 : pandas DataFrame
            The second test dataset

        Returns
        -------
        pandas DataFrame
            The concatenated main test dataset.
        """

        # ANALYSING INDEXES

        has_index1 = False
        has_index2 = False

        if (any([isinstance(x, str) for x in df1.index])):
            has_index1 = True

        if (any([isinstance(x, str) for x in df2.index])):
            has_index2 = True

        # CRUNCHING

        s1 = set(df1.columns)
        s2 = set(df2.columns)
        s = s1 & s2

        ########################################################
        # FIRST CASE : df1 and df2 have exactly the same columns
        ########################################################

        if (s1 == s2):

            return pd.concat([df1, df2])

        else:

        #################################################################
        # SECOND CASE : df2 columns is included in df1 columns (strictly)
        #################################################################

            if (s == s2):
                pass


    def crunch(self, data1, data2):

        """Crunches two different data dictionaries into one. Recursive crunch.

        Parameters
        ----------
        data1 : dict
            The first dataset you want to gather. Contains "train" and "test" keys

        data2 : dict
            The second dataset you want to gather. Contains "train" and "test" keys

        Returns
        -------
        dict
            The reduced main data dictionnary.
        """

        start_time = time.time()

        if (self.verbose):
            sys.stdout.write("crunching your current file with the previous one ...")

        data = {"train" : self.__crunch_train(data1["train"], data2["train"]),
                "test" : self.__crunch_test(data1["test"], data2["test"])}

        if (self.verbose):
            sys.stdout.write(" - " + str(np.round(time.time() - start_time, 2)) + " s")
            print("")

        return data

    def split(self, df, target_name):

        """Splits the dataframe into a training and a test sets.

        IMPORTANT: a dataset is considered as a test set if it does not contain the target value. Otherwise it is
        considered as part of a train set.

        Parameters
        ----------
        df : pandas DataFrame
            The dataset

        target_name : str
            The target name according to which split is performed between train and test

        Returns
        -------
        dict
            Dataset dictionnary containing "train" and "test" keys with the associated dataframes
        """

        start_time = time.time()

        if (self.verbose):
            sys.stdout.write("splitting your current file into train and test sets ...")

        if (target_name in df.columns):

            is_null = df[target_name].isnull()
            data = {"train" : df[~is_null],
                    "test" : df[is_null].drop(target_name, axis=1)}

        else:

            data  = {"train" : pd.DataFrame([], columns = df.columns + [target_name]),
                    "test" : df}

        if (self.verbose):
            sys.stdout.write(" - " + str(np.round(time.time() - start_time, 2)) + " s")
            print("")

        return data

    def read(self, path):

        """Reads data.

        Accepted formats : csv, xls, json and h5

        Parameters
        ----------
        path : str
            The path to the dataset.

        Returns
        -------
        pandas dataframe
            Dataset.
        """

        start_time = time.time()

        # checks

        if (path is None):

            raise ValueError("You must specify the path to load the data")

        else:

            if (type(path) != str):

                raise ValueError("Argument path must be a string !")

            else:

                if ("." not in path):

                    raise ValueError("You need to specify a file format using "
                                     "'.csv' or '.xls' or '.json' or '.h5' !")

        # reading

        type_doc = path.split(".")[-1]

        if (self.verbose):
            sys.stdout.write("reading " + type_doc + " : " + path.split("/")[-1] + " ...")

        try:
            if (type_doc == 'csv'):

                df = pd.read_csv(path, sep=None, engine='python')

            elif (type_doc == 'tsv'):

                df = pd.read_csv(path, sep='\t')

            elif (type_doc == 'xls'):

                df = pd.read_excel(path, header=self.header)

            elif (type_doc == 'h5'):

                df = pd.read_hdf(path)

            elif (type_doc == 'json'):

                df = pd.read_json(path)

            else:

                raise ValueError("The document extension " + type_doc + "cannot be handled")

        except:

            raise ValueError("Cannot read the file !")

        # exceptions

        try:
            del df["Unnamed: 0"]
        except:
            pass

        if (len(set(df.columns)) < len(df.columns)):
            raise ValueError("Your dataset contains duplicate feature names !")

        if (len(df.columns) <= 2):
            warnings.warn("Your dataset contains less than 2 features.")

        # set index

        if (set(df.index) != set(range(df.shape[0]))):
            df.index = [str(i) for i in list(df.index)]

        if (self.verbose):
            sys.stdout.write(" - " + str(np.round(time.time() - start_time, 2)) + " s")
            print("")

        return df.reindex_axis(sorted(df.columns), axis=1)

    def cast(self, df):

        """Casts data to a standard format.

        - casts lists into features
        - tries to cast variables into float
        - casts dates and extracts timestamp from 01/01/2017, year, month, day, day_of_week and hour

        Parameters
        ----------
        df : pandas dataframe
            The input dataset.

        Returns
        -------
        pandas dataframe
            Cleaned dataset.
        """

        start_time = time.time()

        if (self.verbose):
            sys.stdout.write("casting data ...")

        # Casting lists

        df = pd.concat(Parallel(n_jobs=-1)(delayed(convert_list)(df[col]) \
                                           for col in df.columns), axis=1)

        # Casting floats and dates

        df = pd.concat(Parallel(n_jobs=-1)(delayed(convert_float_and_dates) \
                                               (df[col]) for col in df.columns),
                       axis=1)

        if (self.verbose):
            sys.stdout.write(" - " + str(np.round(time.time() - start_time, 2)) + " s")
            print("")

        return df

    def run(self, Lpath, target_name):

        """Runs the following steps :

        - Reading of the files
        - Spliting into train and test datasets
        - Auto Data Crunching
        - Type conversion/cast
        - Task detection

        Parameters
        ----------
        Lpath : list, default = None
            List of str paths to load the data

        target_name : str, default = None
            The name of the target. Works for both classification
            (multiclass or not) and regression.

        Returns
        -------
        dict
            Dictionary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : encoded pandas Serie for the target on train set (with dtype='float' for a regression or dtype='int' for a classification)
        """

        if (type(Lpath) != list):

            raise ValueError("You must specify a list of paths "
                             "to load all the data")

        else:

            if (len(Lpath) == 0):

                raise ValueError("The list of paths is empty !")


        if (self.to_path is None):

            raise ValueError("You must specify a path to save your data "
                             "and make sure your files are not already saved")

        if (type(target_name) != str):

            raise ValueError("Parameter 'target_name' must be a string !")


        ##############################################################
        #                    building the databases
        ##############################################################

        start_time = time.time()

        if (self.verbose):
            print("")
            print("STEP 1 - reading and building the databases ...")
            print("")
            print("")

        ###############################################
        # crunching train sets and test sets separately
        ###############################################

        for i, path in enumerate(Lpath):

            if (i==0):
                data = self.split(self.read(path), target_name)

            else:
                data = self.crunch(data, self.split(self.read(path),
                                                    target_name))

        ###################################################
        # crunching train with test set and dropping target
        ###################################################

        cols = sorted(list(set(data["train"].columns)
                           & set(data["test"].columns)))

        data = {"train" : data["train"][cols],
                "test" : data["test"][cols],
                "target" : data["train"][target_name]}

        if (data["train"].index.nunique() < data["train"].shape[0]):
            data["train"].index = range(data["train"].shape[0])

        if (data["test"].index.nunique() < data["test"].shape[0]):
            data["test"].index = range(data["test"].shape[0])

        if (data["target"].index.nunique() < data["target"].shape[0]):
            data["target"].index = range(data["target"].shape[0])

        if (self.verbose):
            print("")
            print("CPU time for STEP 1: %s seconds" % np.round((time.time() - start_time), 2))
            print("")

        ############
        # Exceptions
        ############

        if (data["train"].shape[0] == 0):
            raise ValueError("You have no train dataset. Please check that the target "
                             "name is correct.")

        if (len(cols) == 0):
            raise ValueError("You have no common features between train and test sets. "
                             "Please check the names of your features.")

        if ((data["test"].shape[0] == 0) & (self.verbose)):
            print("")
            print("You have no test dataset !")

        ##############################################################
        #                    casting the databases
        ##############################################################

        start_time = time.time()

        if (self.verbose):
            print("")
            print("STEP 2 - casting the databases ...")
            print("")
            print("")

        ###################
        # casting train set
        ###################

        if (self.verbose):
            sys.stdout.write("training set : ")

        data["train"] = self.cast(data["train"])

        ##################
        # casting test set
        ##################

        if (self.verbose):
            sys.stdout.write("test set : ")

        data["test"] = self.cast(data["test"])

        # cols can be modified after casting lists...
        cols = sorted(list(set(data["train"].columns)
                           & set(data["test"].columns)))

        data = {"train": data["train"][cols],
              "test": data["test"][cols],
              "target": data["target"]}

        ##################
        # task detection
        ##################

        te = TargetEncoder()
        te.detect_task(data["target"])
        task = te.get_task()

        if (self.verbose):
            print("")
            print("CPU time for STEP 2: %s seconds" % np.round((time.time() - start_time), 2))
            print("")

        ##############################################################
        #                    analysing the databases
        ##############################################################

        start_time = time.time()

        if (self.verbose):
            print("")
            print("STEP 3 - analysing the training set ...")
            print("")
            print("")

        # exceptions
        if (len(cols) == 1):
            warnings.warn("You have only one common feature between train and test sets !")

        Lcat = data["train"].dtypes[data["train"].dtypes == 'object'].index
        Lnum = data["train"].dtypes[data["train"].dtypes != 'object'].index

        self.__stats = dict(data["train"].describe())

        for col in Lcat:
            try:
                self.__stats[col] = data["train"][col].value_counts()
            except:
                pass

        if (self.verbose):
            print("> Number of categorical features: " + str(len(Lcat)))
            print("> Number of numerical features: " + str(len(Lnum)))
            print("> Number of training samples : " + str(data["train"].shape[0]))
            print("> Number of test samples : " + str(data["test"].shape[0]))
            print("")
            print("> Task : " + task)

            if (task == "classification"):
                print("> Target value counts :")
                print("")
                print(data["target"].value_counts())

            else:
                print("> Target description :")
                print("")
                print(data["target"].describe())

            print("")
            print("more information available if calling '.get_stats()' ...")

            print("")
            print("CPU time for STEP 3: %s seconds" % np.round((time.time() - start_time), 2))
            print("")

        ##############################################################
        #                         Dumping
        ##############################################################

        # Creating a folder to save the files and target encoder

        try:
            os.mkdir(self.to_path)
        except OSError:
            pass

        if (self.dump):

            start_time = time.time()

            if (self.verbose):
                print("")
                print("OPTIONAL STEP - saving the databases into directory : " + self.to_path + " ...")
                print("")
                print("")

            ###############
            # dumping train
            ################

            if (self.verbose):
                sys.stdout.write("dumping training set ...")

            start_time_train = time.time()

            # Temp adding target to dump train file...
            data["train"][target_name] = data["target"].values
            data["train"].to_hdf(self.to_path + '/df_train.h5', 'train')
            del data["train"][target_name]

            if (self.verbose):
                sys.stdout.write(" - " + str(np.round(time.time() - start_time_train, 2)) + " s")
                print("")

            ###############
            # dumping test
            ################

            if (self.verbose):
                sys.stdout.write("dumping test set ...")

            start_time_test = time.time()
            data["test"].to_hdf(self.to_path + '/df_test.h5', 'test')

            if (self.verbose):
                sys.stdout.write(" - " + str(np.round(time.time() - start_time_train, 2)) + " s")
                print("")

                print("")
                print("CPU time for OPTIONAL STEP : %s seconds" % (time.time() - start_time_test))
                print("")

        else:
            pass

        return data
