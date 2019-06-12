# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause
"""Test mlbox.preprocessing.reader module."""
import sys

import pytest
import pandas as pd
import numpy as np


from mlbox.preprocessing.reader import convert_list
from mlbox.preprocessing.reader import convert_float_and_dates
from mlbox.preprocessing.reader import Reader


def test_init_reader():
    """Test init method of Reader class."""
    reader = Reader()
    assert not reader.sep
    assert reader.header == 0
    assert not reader.to_hdf5
    assert reader.to_path == "save"
    assert reader.verbose


def test_clean_reader():
    """Test clean method of Reader class."""
    reader = Reader()
    with pytest.raises(ValueError):
        reader.clean(path=None, drop_duplicate=False)
    with pytest.raises(ValueError):
        reader.clean(path="data_for_tests/train.csv")
    reader = Reader(sep=",")
    df = reader.clean(path="data_for_tests/train.csv")
    assert np.shape(df) == (891, 12)
    with pytest.raises(ValueError):
        reader.clean(path="data_for_tests/train.wrong_extension")
    df_drop = reader.clean(path="data_for_tests/train.csv",
                           drop_duplicate=True)
    assert np.shape(df_drop) == (891, 12)
    assert np.all(df["Name"] == df_drop["Name"])
    reader = Reader()
    df_excel = reader.clean(path="data_for_tests/train.xls")
    assert np.shape(df_excel) == (891, 12)
    assert np.all(df["Name"] == df_excel["Name"])
    if sys.version_info[0] >= 3:
        df_hdf = reader.clean(path="data_for_tests/train.h5")
        assert np.shape(df_hdf) == (891, 12)
        assert np.all(df["Name"] == df_hdf["Name"])
    df_json = reader.clean(path="data_for_tests/train.json")
    assert np.shape(df_json) == (891, 12)


def test_train_test_split_reader():
    """Test train_test_split method of Reader class."""
    reader = Reader(sep=",")
    with pytest.raises(ValueError):
        reader.train_test_split(Lpath=None, target_name="target")
    with pytest.raises(ValueError):
        reader.train_test_split(Lpath=["data_for_tests/train.csv"],
                                target_name=None)
    with pytest.raises(ValueError):
        reader = Reader(to_path=None)
        reader.train_test_split(Lpath=["data_for_tests/train.csv"],
                                target_name="Survived")
    reader = Reader(sep=",")
    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv"],
                                   target_name="Survived")
    assert len(dict) == 3
    assert "train" in list(dict.keys())
    assert "test" in list(dict.keys())
    assert "target" in list(dict.keys())
    assert np.all(dict["train"].columns == dict["train"].columns)
    if (sys.version_info[0] >= 3 and sys.platform != "win32"):
        reader = Reader(to_hdf5=True)
        dict = reader.train_test_split(Lpath=["data_for_tests/train.h5"],
                                       target_name="Survived")
        assert len(dict) == 3
        assert "train" in list(dict.keys())
        assert "test" in list(dict.keys())
        assert "target" in list(dict.keys())
        assert np.all(dict["train"].columns == dict["train"].columns)


def test_convert_list_reader():
    """Test convert_list function of reader module."""
    data_list = list()
    data_list.append([1, 2])
    data_list.append([3, 4])
    index = ['a', 'b']
    serie = pd.Series(data=data_list, index=index, name="test")
    df = convert_list(serie)
    assert np.all(df.index == serie.index)
    assert np.all(df.columns.values == ['test_item1', 'test_item2'])


def test_convert_float_and_dates_reader():
    """Test convert_float_and_dates function of reader module."""
    index = ['a', 'b', 'c']
    values = [1, 2, 3]
    serie = pd.Series(data=values, index=index)
    serie = convert_float_and_dates(serie)
    assert serie.dtype == 'float64'

    index = ['a', 'b', 'c']
    values = np.array(['2007-07-13', '2006-01-13', '2010-08-13'],
                      dtype='datetime64')
    serie = pd.Series(data=values,
                      index=index,
                      dtype='datetime64[ns]',
                      name="test")
    df = convert_float_and_dates(serie)
    assert np.all(df.index == serie.index)
    assert np.all(df.columns.values == ['test_TIMESTAMP',
                                        'test_YEAR',
                                        'test_MONTH',
                                        'test_DAY',
                                        'test_DAYOFWEEK',
                                        'test_HOUR'])

    index = ['a', 'b', 'c']
    values = np.array(['2007-07-13', '2006-01-13', '2010-08-13'])
    serie = pd.Series(data=values, index=index, name="test")
    df = convert_float_and_dates(serie)
    assert np.all(df.index == serie.index)
    assert np.all(df.columns.values == ['test_TIMESTAMP',
                                        'test_YEAR',
                                        'test_MONTH',
                                        'test_DAY',
                                        'test_DAYOFWEEK',
                                        'test_HOUR'])
