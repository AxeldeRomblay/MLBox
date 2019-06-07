"""Test mlbox.preprocessing.drift module."""
# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause

import pytest
import pandas as pd

from mlbox.preprocessing.drift import DriftThreshold
from sklearn.tree import DecisionTreeClassifier


def test_init_drift_threshold():
    """Test init method of DriftThreshold class."""
    drift_threshold = DriftThreshold()
    assert drift_threshold.threshold == 0.6
    assert drift_threshold.subsample == 1.
    assert isinstance(drift_threshold.estimator,
                      type(DecisionTreeClassifier()))
    assert drift_threshold.n_folds == 2
    assert drift_threshold.stratify
    assert drift_threshold.random_state == 1
    assert drift_threshold.n_jobs == -1
    assert not drift_threshold._DriftThreshold__fitOK


def test_get_params_drift_threshold():
    """Test get_params method of DriftThreshold class."""
    drift_threshold = DriftThreshold()
    dict = {'threshold': 0.6,
            'subsample': 1.,
            'n_folds': 2,
            'stratify': True,
            'random_state': 1,
            'n_jobs': -1}
    dict_get_params = drift_threshold.get_params()
    assert dict_get_params["threshold"] == dict["threshold"]
    assert dict_get_params["subsample"] == dict["subsample"]
    assert dict_get_params["n_folds"] == dict["n_folds"]
    assert dict_get_params["stratify"] == dict["stratify"]
    assert dict_get_params["random_state"] == dict["random_state"]
    assert dict_get_params["n_jobs"] == dict["n_jobs"]


def test_set_params_drift_threshold():
    """Test set_params method of DriftThreshold class."""
    drift_threshold = DriftThreshold()
    dict = {'threshold': 0.6,
            'subsample': 1.,
            'n_folds': 2,
            'stratify': True,
            'random_state': 1,
            'n_jobs': -1}
    drift_threshold.set_params(**dict)
    dict_get_params = drift_threshold.get_params()
    assert dict_get_params["threshold"] == dict["threshold"]
    assert dict_get_params["subsample"] == dict["subsample"]
    assert dict_get_params["n_folds"] == dict["n_folds"]
    assert dict_get_params["stratify"] == dict["stratify"]
    assert dict_get_params["random_state"] == dict["random_state"]
    assert dict_get_params["n_jobs"] == dict["n_jobs"]


# def test_fit_drift_threshold():
    # df_train = pd.read_csv("data_for_tests/train.csv")
    # df_test = pd.read_csv("data_for_tests/test.csv")
    # # df = df.rename(columns={'Survived': 'newName1', 'oldName2': 'newName2'})
    # drift_estimator = DriftEstimator()
    # drift_estimator.fit(df_train, df_test)
    # assert drift_estimator._DriftEstimator__fitOK
#     encoder = Categorical_encoder(strategy="wrong_strategy")
#     with pytest.raises(ValueError):
#         encoder.fit(df, df["Survived"])
#     encoder.set_params(strategy="label_encoding")
#     encoder.fit(df, df["Survived"])
#     assert encoder._Categorical_encoder__fitOK
#     encoder.set_params(strategy="dummification")
#     encoder.fit(df, df["Survived"])
#     assert encoder._Categorical_encoder__fitOK
#     encoder.set_params(strategy="random_projection")
#     encoder.fit(df, df["Survived"])
#     assert encoder._Categorical_encoder__fitOK
#
#
# def test_transform_encoder():
#     df = pd.read_csv("data_for_tests/train.csv")
#     encoder = Categorical_encoder()
#     with pytest.raises(ValueError):
#         encoder.transform(df)
#     encoder.fit(df, df["Survived"])
#     df_encoded = encoder.transform(df)
#     assert (df.columns == df_encoded.columns).all()
