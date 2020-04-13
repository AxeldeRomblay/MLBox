# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause

import pandas as pd
from sklearn.preprocessing import LabelEncoder

class TargetEncoder():

    def __init__(self):

        self.__enc = None
        self.__task = None

    def detect_task(self, y):

        if (type(y) != pd.core.series.Series):
            raise ValueError("Target must be a pandas Series")

        else:

            if (y.nunique() <= 2):
                self.__task = "classification"

            else:

                if (y.dtype == object):
                    self.__task = "classification"

                else:

                    try:
                        y = y.apply(float)
                        self.__task = "regression"
                    except:
                        raise ValueError("Target is weird. Cannot be handled with classification or regression !")

        return self

    def get_task(self):

        return self.__task

    def get_encoder(self):

        return self.__enc

    def fit_transform(self, y):

        if (self.get_task() == None):
            self.detect_task(y)
        else:
            pass

        if (self.get_task() == "regression"):
            return y

        else:
            self.__enc = LabelEncoder()
            return  pd.Series(self.__enc.fit_transform(y.values),
                              index=y.index,
                              name=y.name,
                              dtype='int')




