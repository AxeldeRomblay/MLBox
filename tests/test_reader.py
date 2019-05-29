#!/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
import pytest
import pandas as pd
import numpy as np

from mlbox.preprocessing.reader import convert_list
from mlbox.preprocessing.reader import convert_float_and_dates
from mlbox.preprocessing.reader import Reader


def test_init_reader():
    reader = Reader()
    assert not reader.sep
    assert reader.header == 0
    assert not reader.to_hdf5
    assert reader.to_path == "save"
    assert reader.verbose


def test_clean():
    reader = Reader()
    with pytest.raises(ValueError):
        reader.clean(path=None, drop_duplicate=False)
    with pytest.raises(ValueError):
        reader.clean(path="data_for_tests/train.csv")
    reader = Reader(sep=",")
    df = reader.clean(path="data_for_tests/train.csv")
    with pytest.raises(ValueError):
        reader.clean(path="data_for_tests/train.wrong_extension")
    df = reader.clean(path="data_for_tests/train.csv", drop_duplicate=True)


def test_train_test_split():
    reader = Reader(sep=",")
    with pytest.raises(ValueError):
        reader.train_test_split(Lpath=None, target_name="target")
    with pytest.raises(ValueError):
        reader.train_test_split(Lpath=["data_for_tests/train.csv"], target_name=None)
    with pytest.raises(ValueError):
        reader = Reader(to_path=None)
        reader.train_test_split(Lpath=["data_for_tests/train.csv"], target_name="Survived")
    reader = Reader(sep=",")
    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv"], target_name="Survived")


def test_convert_list():
    data_list = list()
    data_list.append([1, 2])
    data_list.append([3, 4])
    index = ['a', 'b']
    serie = pd.Series(data=data_list, index=index, name="test")
    df = convert_list(serie)
    assert np.all(df.index == serie.index)
    assert np.all(df.columns.values == ['test_item1', 'test_item2'])


def test_convert_float_and_dates():
    index = ['a', 'b', 'c']
    values = [1, 2, 3]
    serie = pd.Series(data=values, index=index)
    serie = convert_float_and_dates(serie)
    assert serie.dtype == 'float64'

    index = ['a', 'b', 'c']
    values = np.array(['2007-07-13', '2006-01-13', '2010-08-13'], dtype='datetime64')
    serie = pd.Series(data=values, index=index, dtype='datetime64[ns]', name="test")
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
