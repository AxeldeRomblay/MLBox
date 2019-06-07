"""Test mlbox.preprocessing.drift module."""
# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause

import pytest
import pandas as pd

from mlbox.preprocessing.drift import DriftThreshold
from mlbox.preprocessing.drift import sync_fit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


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
            'estimator': DecisionTreeClassifier(max_depth=6),
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


def test_fit_drift_threshold():
    """Test fit method of DriftThreshold class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    drift_threshold = DriftThreshold()
    drift_threshold.fit(df_train, df_test)
    assert drift_threshold._DriftThreshold__fitOK


def test_transform_drift_threshold():
    """Test transform method of DriftThreshold class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    drift_threshold = DriftThreshold()
    with pytest.raises(ValueError):
        drift_threshold.transform(df_train)
    drift_threshold.fit(df_train, df_test)
    df_transformed = drift_threshold.transform(df_train)
    assert (df_train.columns == df_transformed.columns).all()


def test_get_support_drift_threshold():
    """Test get_support method of DriftThreshold class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    drift_threshold = DriftThreshold()
    with pytest.raises(ValueError):
        drift_threshold.get_support()
    drift_threshold.fit(df_train, df_test)
    keep_list = drift_threshold.get_support()
    drop_list = drift_threshold.get_support(complement=True)
    for name in ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp']:
        assert (name in keep_list)
    assert not drop_list


def test_drifts_drift_threshold():
    """Test drifts method of DriftThreshold class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    drift_threshold = DriftThreshold()
    with pytest.raises(ValueError):
        drift_threshold.drifts()
    drift_threshold.fit(df_train, df_test)
    drifts = drift_threshold.drifts()
    for name in ['Age', 'Fare', 'Parch', 'Pclass', 'SibSp']:
        assert (name in list(drifts.keys()))


def test_sync_fit_drift_threshold():
    """Test method sync_fit of drift_threshold module."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    estimator = RandomForestClassifier(n_estimators=50,
                                       n_jobs=-1,
                                       max_features=1.,
                                       min_samples_leaf=5,
                                       max_depth=5)

    score = sync_fit(df_train, df_test, estimator)
    assert 0 <= score
