# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause
"""Test mlbox.model.regression.stacking_regressor module."""
import pytest
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from mlbox.model.regression.stacking_regressor import StackingRegressor


def test_init_stacking_regressor():
    """Test init method of StackingRegressor class."""
    with pytest.raises(ValueError):
        stacking_regressor = StackingRegressor(base_estimators=dict())
    with pytest.raises(ValueError):
        stacking_regressor = StackingRegressor(n_folds=dict())
    with pytest.raises(ValueError):
        stacking_regressor = StackingRegressor(copy="True")
    with pytest.raises(ValueError):
        stacking_regressor = StackingRegressor(random_state="1")
    with pytest.raises(ValueError):
        stacking_regressor = StackingRegressor(verbose="True")
    stacking_regressor = StackingRegressor()
    assert len(stacking_regressor.base_estimators) == 3
    assert isinstance(stacking_regressor.level_estimator,
                      type(LinearRegression()))
    assert stacking_regressor.n_folds == 5
    assert not stacking_regressor.copy
    assert stacking_regressor.random_state == 1
    assert stacking_regressor.verbose
    assert not stacking_regressor._StackingRegressor__fitOK
    assert not stacking_regressor._StackingRegressor__fittransformOK


def test_get_params_stacking_regressor():
    """Test get_params method of StackingRegressor class."""
    stacking_regressor = StackingRegressor()
    dict = stacking_regressor.get_params()
    assert len(dict["base_estimators"]) == 3
    assert isinstance(dict["level_estimator"],
                      type(LinearRegression()))
    assert dict["n_folds"] == 5
    assert not dict["copy"]
    assert dict["random_state"] == 1
    assert dict["verbose"]


def test_set_params_stacking_regressor():
    """Test set_params method of StackingRegressor class."""
    stacking_regressor = StackingRegressor()
    stacking_regressor.set_params(n_folds=6)
    assert stacking_regressor.n_folds == 6
    stacking_regressor.set_params(copy=True)
    assert stacking_regressor.copy
    stacking_regressor.set_params(random_state=2)
    assert stacking_regressor.random_state == 2
    stacking_regressor.set_params(verbose=False)
    assert not stacking_regressor.verbose
    with pytest.warns(UserWarning) as record:
        stacking_regressor.set_params(wrong_parameters=None)
    assert len(record) == 1


def test_fit_transform_stacking_regressor():
    """Test fit_transform method of Stacking regressor class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    stacking_regressor = StackingRegressor()
    with pytest.raises(ValueError):
        stacking_regressor.fit_transform(None, y_train)
    with pytest.raises(ValueError):
        stacking_regressor.fit_transform(df_train, None)
    stacking_regressor.fit_transform(df_train, y_train)
    assert stacking_regressor._StackingRegressor__fittransformOK


def test_transform_stacking_regressor():
    """Test transform method of StackingRegressor class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    stacking_regressor = StackingRegressor()
    with pytest.raises(ValueError):
        stacking_regressor.transform(None)
    with pytest.raises(ValueError):
        stacking_regressor.transform(df_test)
    stacking_regressor.fit_transform(df_train, y_train)
    results = stacking_regressor.transform(df_test)
    assert len(results.columns == 3)


def test_fit_stacking_regressor():
    """Test fit method of StackingRegressor class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    stacking_regressor = StackingRegressor(verbose=True)
    stacking_regressor.fit(df_train, y_train)
    assert stacking_regressor._StackingRegressor__fitOK


def test_predict_stacking_regressor():
    """Test predict method of StackingRegressor class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    stacking_regressor = StackingRegressor()
    with pytest.raises(ValueError):
        stacking_regressor.predict(df_test)
    stacking_regressor.fit(df_train, y_train)
    results = stacking_regressor.predict(df_test)
    assert np.shape(results) == (418,)
