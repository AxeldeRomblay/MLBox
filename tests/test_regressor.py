#!/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
# import pytest

import pytest
import pandas as pd
import numpy as np

from mlbox.model.regression.regressor import Regressor
from mlbox.encoding.categorical_encoder import Categorical_encoder
from lightgbm import LGBMRegressor


def test_init_regressor():
    regressor = Regressor()
    assert regressor._Regressor__strategy == "LightGBM"
    assert regressor._Regressor__regress_params == {}
    assert regressor._Regressor__regressor
    assert not regressor._Regressor__col
    assert not regressor._Regressor__fitOK


def test_get_params_regressor():
    regressor = Regressor()
    params = regressor.get_params()
    assert params == {'strategy': "LightGBM"}
    assert not regressor._Regressor__regress_params


def test_set_params_regressor():
    regressor = Regressor()
    regressor.set_params(strategy="LightGBM")
    assert regressor._Regressor__strategy == "LightGBM"
    regressor.set_params(strategy="RandomForest")
    assert regressor._Regressor__strategy == "RandomForest"
    regressor.set_params(strategy="ExtraTrees")
    assert regressor._Regressor__strategy == "ExtraTrees"
    regressor.set_params(strategy="RandomForest")
    assert regressor._Regressor__strategy == "RandomForest"
    regressor.set_params(strategy="Tree")
    assert regressor._Regressor__strategy == "Tree"
    regressor.set_params(strategy="AdaBoost")
    assert regressor._Regressor__strategy == "AdaBoost"
    regressor.set_params(strategy="Linear")
    assert regressor._Regressor__strategy == "Linear"
    regressor.set_params(strategy="Bagging")
    assert regressor._Regressor__strategy == "Bagging"
    with pytest.warns(UserWarning) as record:
        regressor.set_params(wrong_strategy="wrong_strategy")
    assert len(record) == 1


def test_set_regressor():
    regressor = Regressor()
    with pytest.raises(ValueError):
        regressor._Regressor__set_regressor("wrong_strategy")


def test_fit_regressor():
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    regressor = Regressor()
    regressor.fit(df_train, y_train)
    assert np.all(regressor._Regressor__col == df_train.columns)
    assert regressor._Regressor__fitOK

def test_feature_importances_regressor():
    regressor = Regressor()
    with pytest.raises(ValueError):
        regressor.feature_importances()
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    regressor.set_params(strategy="Linear")
    regressor.fit(df_train, y_train)
    importance = regressor.feature_importances()
    assert importance != {}
    regressor.set_params(strategy="RandomForest")
    regressor.fit(df_train, y_train)
    importance = regressor.feature_importances()
    assert importance != {}
    regressor.set_params(strategy="AdaBoost")
    regressor.fit(df_train, y_train)
    importance = regressor.feature_importances()
    assert importance != {}
    # regressor.set_params(strategy="Bagging")
    # regressor.fit(df_train, y_train)
    # importance = regressor.feature_importances()
    # assert importance != {}


def test_predict_regressor():
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    regressor = Regressor()
    with pytest.raises(ValueError):
        regressor.predict(df_train)
    regressor.fit(df_train, y_train)
    with pytest.raises(ValueError):
        regressor.predict(None)
    assert len(regressor.predict(df_train)) > 0


# def test_transform_regressor():
#     df_train = pd.read_csv("data_for_tests/clean_train.csv")
#     y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
#     regressor = Regressor(strategy="Linear")
#     with pytest.raises(ValueError):
#         regressor.transform(df_train)
#     regressor.fit(df_train, y_train)
#     with pytest.raises(ValueError):
#         regressor.transform(None)
#     assert len(regressor.transform(df_train)) > 0


def test_score_regressor():
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    regressor = Regressor(strategy="Linear")
    with pytest.raises(ValueError):
        regressor.score(df_train, y_train)
    regressor.fit(df_train, y_train)
    with pytest.raises(ValueError):
        regressor.score(None, y_train)
    with pytest.raises(ValueError):
        regressor.score(df_train, None)
    assert regressor.score(df_train, y_train) > 0

def test_get_estimator():
    regressor = Regressor()
    estimator = regressor.get_estimator()
    assert type(estimator) == type(LGBMRegressor())
