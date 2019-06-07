"""Test mlbox.model.classification.feature_selector module."""
# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause
# import pytest

import pytest
import pandas as pd

from mlbox.model.classification.feature_selector import Clf_feature_selector


def test_init_Clf_feature_selector():
    """Test init method of Clf_feature_selector class."""
    feature_selector = Clf_feature_selector()
    assert feature_selector.strategy == "l1"
    assert feature_selector.threshold == 0.3
    assert not feature_selector._Clf_feature_selector__fitOK
    assert feature_selector._Clf_feature_selector__to_discard == []


def test_get_params_Clf_feature_selector():
    """Test get_params method of Clf_feature_selector class."""
    feature_selector = Clf_feature_selector()
    dict = {'strategy': "l1",
            'threshold': 0.3}
    assert feature_selector.get_params() == dict


def test_set_params_Clf_feature_selector():
    """Test set_params method of Clf_feature_selector class."""
    feature_selector = Clf_feature_selector()
    feature_selector.set_params(strategy="variance")
    assert feature_selector.strategy == "variance"
    feature_selector.set_params(threshold=0.2)
    assert feature_selector.threshold == 0.2
    with pytest.warns(UserWarning) as record:
        feature_selector.set_params(wrong_strategy="wrong_strategy")
    assert len(record) == 1


def test_fit_Clf_feature_selector():
    """Test fit method of Clf_feature_selector class."""
    feature_selector = Clf_feature_selector()
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    with pytest.raises(ValueError):
        feature_selector.fit(None, y_train)
    with pytest.raises(ValueError):
        feature_selector.fit(df_train, None)
    feature_selector.fit(df_train, y_train)
    assert feature_selector._Clf_feature_selector__fitOK
    feature_selector.set_params(strategy="variance")
    feature_selector.fit(df_train, y_train)
    assert feature_selector._Clf_feature_selector__fitOK
    feature_selector.set_params(strategy="rf_feature_importance")
    feature_selector.fit(df_train, y_train)
    assert feature_selector._Clf_feature_selector__fitOK
    feature_selector.set_params(strategy="wrond_strategy")
    with pytest.raises(ValueError):
        feature_selector.fit(df_train, y_train)


def test_transform_Clf_feature_selector():
    """Test transform method of Clf_feature_selector class."""
    feature_selector = Clf_feature_selector(threshold=0)
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    with pytest.raises(ValueError):
        feature_selector.transform(df_train)
    feature_selector.fit(df_train, y_train)
    with pytest.raises(ValueError):
        feature_selector.transform(None)
    df_transformed = feature_selector.transform(df_train)
    assert (df_transformed.columns == df_train.columns).all()


def test_fit_transform_Clf_feature_selector():
    """Test fit_transform method of Clf_feature_selector class."""
    feature_selector = Clf_feature_selector(threshold=0)
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    df_transformed = feature_selector.fit_transform(df_train, y_train)
    assert (df_transformed.columns == df_train.columns).all()
