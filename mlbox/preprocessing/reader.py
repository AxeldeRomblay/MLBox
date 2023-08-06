# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
import sys
import pickle
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed


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
        df[serie.name + "_TIMESTAMP"] = (pandas.DatetimeIndex(serie) -
                                         pandas.datetime(2017, 1, 1)
                                         ).total_seconds()

        df[serie.name + "_YEAR"] = pandas.DatetimeIndex(serie).year.astype(  # noqa
            float)  # TODO: be careful with nan ! object or float ??

        df[serie.name + "_MONTH"] = pandas.DatetimeIndex(serie).month.astype(  # noqa
            float)  # TODO: be careful with nan ! object or float ??

        df[serie.name + "_DAY"] = pandas.DatetimeIndex(serie).day.astype(
            float)  # TODO: be careful with nan ! object or float ??

        df[serie.name + "_DAYOFWEEK"] = pandas.DatetimeIndex(serie).dayofweek.astype(  # noqa
            float)  # TODO: be careful with nan ! object or float ??

        df[serie.name + "_HOUR"] = pandas.DatetimeIndex(serie).hour.astype(float) + \
                                   pandas.DatetimeIndex(serie).minute.astype(float)/60. + \
                                   pandas.DatetimeIndex(serie).second.astype(float)/3600.

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

                df[serie.name + "_TIMESTAMP"] = (serie_to_df -
                                                 pandas.datetime(2017, 1, 1)
                                                 ).total_seconds()

                df[serie.name + "_YEAR"] = serie_to_df.year.astype(
                    float)  # TODO: be careful with nan ! object or float??

                df[serie.name + "_MONTH"] = serie_to_df.month.astype(
                    float)  # TODO: be careful with nan ! object or float??

                df[serie.name + "_DAY"] = serie_to_df.day.astype(
                    float)  # TODO: be careful with nan ! object or float??

                df[serie.name + "_DAYOFWEEK"] = serie_to_df.dayofweek.astype(
                    float)  # TODO: be careful with nan ! object or float??

                df[serie.name + "_HOUR"] = serie_to_df.hour.astype(float) + \
                                           serie_to_df.minute.astype(float)/60. + \
                                           serie_to_df.second.astype(float) / 3600.

                return df

            except:

                return serie


class Reader():

    """Reads and cleans data

    Parameters
    ----------
    sep : str, defaut = None
         Delimiter to use when reading a csv file.

    header : int or None, default = 0.
        If header=0, the first line is considered as a header.
        Otherwise, there is no header.
        Useful for csv and xls files.

    to_hdf5 : bool, default = True
        If True, dumps each file to hdf5 format.

    to_path : str, default = "save"
        Name of the folder where files and encoders are saved.

    verbose : bool, defaut = True
        Verbose mode
    """

    def __init__(self,
                 sep=None,
                 header=0,
                 to_hdf5=False,
                 to_path="save",
                 verbose=True):

        self.sep = sep
        self.header = header
        self.to_hdf5 = to_hdf5
        self.to_path = to_path
        self.verbose = verbose

    def clean(self, path, drop_duplicate=False):

        """Reads and cleans data (accepted formats : csv, xls, json and h5):

        - del Unnamed columns
        - casts lists into variables
        - try to cast variables into float
        - cleans dates and extracts timestamp from 01/01/2017, year, month, day, day_of_week and hour
        - drop duplicates (if drop_duplicate=True)

        Parameters
        ----------
        path : str
            The path to the dataset.

        drop_duplicate: bool, default = False
            If True, drop duplicates when reading each file.

        Returns
        -------
        pandas dataframe
            Cleaned dataset.
        """

        ##############################################################
        #                           Reading
        ##############################################################

        start_time = time.time()

        if (path is None):

            raise ValueError("You must specify the path to load the data")

        else:

            type_doc = path.split(".")[-1]

            if (type_doc == 'csv'):

                if (self.sep is None):
                    raise ValueError("You must specify the separator "
                                     "for a csv file")
                else:
                    if (self.verbose):
                        print("")
                        print("reading csv : " + path.split("/")[-1] + " ...")
                    df = pd.read_csv(path,
                                     sep=self.sep,
                                     header=self.header,
                                     engine='c',
                                     error_bad_lines=False)

            elif (type_doc == 'xls'):

                if (self.verbose):
                    print("")
                    print("reading xls : " + path.split("/")[-1] + " ...")
                df = pd.read_excel(path, header=self.header)

            elif (type_doc == 'h5'):
                if (sys.platform == "win32" and sys.version_info[0] <=3 and sys.version_info[1] <=5):
                    raise ValueError("h5 format not supported for python under 3.6 on windows. Please upgrade python")
                if (self.verbose):
                    print("")
                    print("reading hdf5 : " + path.split("/")[-1] + " ...")

                df = pd.read_hdf(path)

            elif (type_doc == 'json'):
                if (sys.platform == "win32" and sys.version_info[0] <=3 and sys.version_info[1] <=5):
                    raise ValueError("json format not supported for python under 3.6 on windows. Please upgrade python")
                if (self.verbose):
                    print("")
                    print("reading json : " + path.split("/")[-1] + " ...")

                df = pd.read_json(path)

            else:

                raise ValueError("The document extension cannot be handled")

        # Deleting unknown column

        try:
            del df["Unnamed: 0"]
        except:
            pass

        ##############################################################
        #             Cleaning lists, floats and dates
        ##############################################################

        if (self.verbose):
            print("cleaning data ...")

        if (sys.platform == "win32"):
            df = pd.concat([convert_list(df[col]) for col in df.columns], axis=1)
            df = pd.concat([convert_float_and_dates(df[col]) for col in df.columns], axis=1)
        else:
            df = pd.concat(Parallel(n_jobs=-1)(delayed(convert_list)(df[col]) for col in df.columns),
                           axis=1)

            df = pd.concat(Parallel(n_jobs=-1)(delayed(convert_float_and_dates)(df[col]) for col in df.columns),
                           axis=1)

        # Drop duplicates

        if (drop_duplicate):
            if (self.verbose):
                print("dropping duplicates")
            df = df.drop_duplicates()
        else:
            pass

        if (self.verbose):
            print("CPU time: %s seconds" % (time.time() - start_time))

        return df

    def train_test_split(self, Lpath, target_name):

        """Creates train and test datasets

        Given a list of several paths and a target name, automatically creates and cleans train and test datasets.
        IMPORTANT: a dataset is considered as a test set if it does not contain the target value. Otherwise it is
        considered as part of a train set.
        Also determines the task and encodes the target (classification problem only).

        Finally dumps the datasets to hdf5, and eventually the target encoder.

        Parameters
        ----------
        Lpath : list, defaut = None
            List of str paths to load the data

        target_name : str, default = None
            The name of the target. Works for both classification
            (multiclass or not) and regression.

        Returns
        -------
        dict
            Dictionnary containing :

            - 'train' : pandas dataframe for train dataset
            - 'test' : pandas dataframe for test dataset
            - 'target' : encoded pandas Serie for the target on train set (with dtype='float' for a regression or dtype='int' for a classification)

        """

        col = []
        col_train = []
        col_test = []
        df_train = dict()
        df_test = dict()
        y_train = dict()

        if (type(Lpath) != list):

            raise ValueError("You must specify a list of paths "
                             "to load all the data")

        elif (self.to_path is None):

            raise ValueError("You must specify a path to save your data "
                             "and make sure your files are not already saved")



        ##############################################################
        #                    Reading the files
        ##############################################################

        for path in Lpath:

            # Reading each file

            df = self.clean(path, drop_duplicate=False)

            # Checking if the target exists to split into test and train

            if (target_name in df.columns):

                is_null = df[target_name].isnull()

                df_train[path] = df[~is_null].drop(target_name, axis=1)
                df_test[path] = df[is_null].drop(target_name, axis=1)
                y_train[path] = df[target_name][~is_null]

            else:

                df_test[path] = df

        del df

        # Exceptions

        if (sum([df_train[path].shape[0]
                    for path in df_train.keys()]) == 0):
            raise ValueError("You have no train dataset. "
                                "Please check that the "
                                "target name is correct.")

        if ((sum([df_test[path].shape[0]
                    for path in df_test.keys()]) == 0) & (self.verbose)):
            print("")
            print("You have no test dataset !")

        # Finding the common subset of features

        for i, df in enumerate(df_train.values()):

            if (i == 0):
                col_train = df.columns
            else:
                col_train = list(set(col_train) & set(df.columns))

        for i, df in enumerate(df_test.values()):

            if (i == 0):
                col_test = df.columns
            else:
                col_test = list(set(col_test) & set(df.columns))

        # Subset of common features

        col = sorted(list(set(col_train) & set(col_test)))

        if (self.verbose):
            print("")
            print("> Number of common features : " + str(len(col)))

            ##############################################################
            #          Creating train, test and target dataframes
            ##############################################################

            print("")
            print("gathering and crunching for train and test datasets ...")

        # TODO: Optimize
        df_train = pd.concat([df[col] for df in df_train.values()])
        df_test = pd.concat([df[col] for df in df_test.values()])
        y_train = pd.concat([y for y in y_train.values()])  # optimiser !!

        # Checking shape of the target

        if (type(y_train) == pd.core.frame.DataFrame):
            raise ValueError("Your target contains more than two columns !"
                                " Please check that only one column "
                                "is named " + target_name)

        else:
            pass

        # Handling indices

        if (self.verbose):
            print("reindexing for train and test datasets ...")

        if (df_train.index.nunique() < df_train.shape[0]):
            df_train.index = range(df_train.shape[0])

        if (df_test.index.nunique() < df_test.shape[0]):
            df_test.index = range(df_test.shape[0])

        if (y_train.index.nunique() < y_train.shape[0]):
            y_train.index = range(y_train.shape[0])

        # Dropping duplicates

        if (self.verbose):
            print("dropping training duplicates ...")

        # Temp adding target to check (x,y) duplicates...
        df_train[target_name] = y_train.values
        df_train = df_train.drop_duplicates()
        del df_train[target_name]
        y_train = y_train.loc[df_train.index]  # TODO: Need to reindex ?

        # Deleting constant variables

        if (self.verbose):
            print("dropping constant variables on training set ...")
        for var in col:
            if (df_train[var].nunique(dropna=False) == 1):
                del df_train[var]
                del df_test[var]

        # Missing values

        sparse_features = (df_train.isnull().sum() *
                            100. / df_train.shape[0]
                            ).sort_values(ascending=False)
        sparse = True
        if(sparse_features.max() == 0.0):
            sparse = False

        # Print information

        if (self.verbose):
            print("")
            print("> Number of categorical features:"
                    " " + str(len(df_train.dtypes[df_train.dtypes == 'object'].index)))  # noqa
            print("> Number of numerical features:"
                    " " + str(len(df_train.dtypes[df_train.dtypes != 'object'].index)))  # noqa
            print("> Number of training samples : " + str(df_train.shape[0]))
            print("> Number of test samples : " + str(df_test.shape[0]))

            if(sparse):
                print("")
                print("> Top sparse features "
                        "(% missing values on train set):")
                print(np.round(sparse_features[sparse_features > 0.0][:5],
                                1))

            else:
                print("")
                print("> You have no missing values on train set...")

        ##############################################################
        #                    Encoding target
        ##############################################################

        task = "regression"
        count = y_train.nunique()

        if (count <= 2):
            task = "classification"

        else:
            if (y_train.dtype == object):
                task = "classification"
            else:
                # no needs to convert into float
                pass

        if (self.verbose):
            print("")
            print("> Task : " + task)

        if (task == "classification"):
            if (self.verbose):
                print(y_train.value_counts())
                print("")
                print("encoding target ...")
            enc = LabelEncoder()
            y_train = pd.Series(enc.fit_transform(y_train.values),
                                index=y_train.index,
                                name=target_name,
                                dtype='int')

            if count == 1:
                warnings.warn("Your target set has only one class ! Please check it is correct, "
                                "otherwise there is no need to use MLBox...")

        else:
            if (self.verbose):
                print(y_train.describe())

        ##############################################################
        #                         Dumping
        ##############################################################

        # Creating a folder to save the files and target encoder

        try:
            os.mkdir(self.to_path)
        except OSError:
            pass

        if (self.to_hdf5):

            start_time = time.time()

            if (self.verbose):
                print("")
                print("dumping files into directory : " + self.to_path)

            # Temp adding target to dump train file...
            df_train[target_name] = y_train.values
            df_train.to_hdf(self.to_path + '/df_train.h5', 'train')
            del df_train[target_name]

            if (self.verbose):
                print("train dumped")

            df_test.to_hdf(self.to_path + '/df_test.h5', 'test')

            if (self.verbose):
                print("test dumped")
                print("CPU time: %s seconds" % (time.time() - start_time))

        else:
            pass

        if (task == "classification"):
            fhand = open(self.to_path + '/target_encoder.obj', 'wb')
            pickle.dump(enc, fhand)
            fhand.close()
        else:
            pass

        return {"train": df_train,
                "test": df_test,
                'target': y_train}
