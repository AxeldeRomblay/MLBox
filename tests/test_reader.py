#!/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
import pytest
import pandas as pd

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
