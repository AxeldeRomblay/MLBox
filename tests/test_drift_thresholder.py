#!/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
import pytest
import pandas as pd

from mlbox.preprocessing.drift_thresholder import Drift_thresholder
from mlbox.preprocessing.reader import Reader


def test_init_drift_thresholder():
    drift_thresholder = Drift_thresholder()
    assert drift_thresholder.threshold == 0.6
    assert not drift_thresholder.inplace
    assert drift_thresholder.verbose
    assert drift_thresholder.to_path == "save"
    assert drift_thresholder._Drift_thresholder__Ddrifts == {}
    assert not drift_thresholder._Drift_thresholder__fitOK


def test_fit_transform():
    drift_thresholder = Drift_thresholder()
    reader = Reader(sep=",")
    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv"],
                                   target_name="Survived")
    drift_thresholder.fit_transform(dict)
    assert not drift_thresholder._Drift_thresholder__fitOK
    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv",
                                          "data_for_tests/test.csv"],
                                   target_name="Survived")
    drift_thresholder.fit_transform(dict)
    assert drift_thresholder._Drift_thresholder__fitOK
    dict = reader.train_test_split(Lpath=["data_for_tests/inplace_train.csv",
                                          "data_for_tests/inplace_test.csv"],
                                   target_name="Survived")
    drift_thresholder.inplace = True
    drift_thresholder.fit_transform(dict)
    assert drift_thresholder._Drift_thresholder__fitOK



def test_drifts():
    drift_thresholder = Drift_thresholder()
    with pytest.raises(ValueError):
        drift_thresholder.drifts()
    reader = Reader(sep=",")
    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv",
                                          "data_for_tests/test.csv"],
                                   target_name="Survived")
    drift_thresholder.fit_transform(dict)
    drifts = drift_thresholder.drifts()
    assert drifts != {}
