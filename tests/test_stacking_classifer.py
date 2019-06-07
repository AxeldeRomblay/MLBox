# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause
"""Test mlbox.model.classification.stacking_classifier module."""
import pytest
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from mlbox.model.classification.stacking_classifier import StackingClassifier


def test_init_stacking_classifier():
    """Test init method of StackingClassifier class."""
    with pytest.raises(ValueError):
        stacking_classifier = StackingClassifier(base_estimators=dict())
    with pytest.raises(ValueError):
        stacking_classifier = StackingClassifier(n_folds=dict())
    with pytest.raises(ValueError):
        stacking_classifier = StackingClassifier(copy="True")
    with pytest.raises(ValueError):
        stacking_classifier = StackingClassifier(drop_first="True")
    with pytest.raises(ValueError):
        stacking_classifier = StackingClassifier(random_state="1")
    with pytest.raises(ValueError):
        stacking_classifier = StackingClassifier(verbose="True")
    stacking_classifier = StackingClassifier()
    assert len(stacking_classifier.base_estimators) == 3
    assert isinstance(stacking_classifier.level_estimator,
                      type(LogisticRegression()))
    assert stacking_classifier.n_folds == 5
    assert not stacking_classifier.copy
    assert stacking_classifier.drop_first
    assert stacking_classifier.random_state == 1
    assert stacking_classifier.verbose
    assert not stacking_classifier._StackingClassifier__fitOK
    assert not stacking_classifier._StackingClassifier__fittransformOK


def test_get_params_stacking_classifier():
    """Test get_params method StackingClassifier class."""
    stacking_classifier = StackingClassifier()
    dict = stacking_classifier.get_params()
    assert len(dict["base_estimators"]) == 3
    assert isinstance(dict["level_estimator"],
                      type(LogisticRegression()))
    assert dict["n_folds"] == 5
    assert not dict["copy"]
    assert dict["drop_first"]
    assert dict["random_state"] == 1
    assert dict["verbose"]


def test_set_params_stacking_classifier():
    """Test set_params method of StackingClassifier class."""
    stacking_classifier = StackingClassifier()
    stacking_classifier.set_params(n_folds=6)
    assert stacking_classifier.n_folds == 6
    stacking_classifier.set_params(copy=True)
    assert stacking_classifier.copy
    stacking_classifier.set_params(drop_first=False)
    assert not stacking_classifier.drop_first
    stacking_classifier.set_params(random_state=2)
    assert stacking_classifier.random_state == 2
    stacking_classifier.set_params(verbose=False)
    assert not stacking_classifier.verbose
    with pytest.warns(UserWarning) as record:
        stacking_classifier.set_params(wrong_parameters=None)
    assert len(record) == 1


def test_fit_transform_stacking_classifier():
    """Test fit_transform method of StackingClassifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    stacking_classifier = StackingClassifier()
    with pytest.raises(ValueError):
        stacking_classifier.fit_transform(None, y_train)
    with pytest.raises(ValueError):
        stacking_classifier.fit_transform(df_train, None)
    stacking_classifier.fit_transform(df_train, y_train)
    assert stacking_classifier._StackingClassifier__fittransformOK


def test_transform_stacking_classifier():
    """Test transform method of StackingClassifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    stacking_classifier = StackingClassifier()
    with pytest.raises(ValueError):
        stacking_classifier.transform(None)
    with pytest.raises(ValueError):
        stacking_classifier.transform(df_test)
    stacking_classifier.fit_transform(df_train, y_train)
    results = stacking_classifier.transform(df_test)
    assert len(results.columns == 3)


def test_fit_stacking_classifier():
    """Test fit method of StackingClassifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    stacking_classifier = StackingClassifier(verbose=True)
    stacking_classifier.fit(df_train, y_train)
    assert stacking_classifier._StackingClassifier__fitOK


def test_predict_proba_stacking_classifier():
    """Test predict_proba method of StackingClassifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    stacking_classifier = StackingClassifier()
    with pytest.raises(ValueError):
        stacking_classifier.predict_proba(df_test)
    stacking_classifier.fit(df_train, y_train)
    results = stacking_classifier.predict_proba(df_test)
    assert np.shape(results) == (418, 2)


def test_predict_stacking_classifier():
    """Test predict method of StackingClassifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    df_test = pd.read_csv("data_for_tests/clean_test.csv")
    stacking_classifier = StackingClassifier()
    with pytest.raises(ValueError):
        stacking_classifier.predict(df_test)
    stacking_classifier.fit(df_train, y_train)
    results = stacking_classifier.predict(df_test)
    assert np.shape(results) == (418,)
