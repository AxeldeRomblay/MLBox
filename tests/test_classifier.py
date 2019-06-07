"""Test mlbox.model.classification.classifier module."""
# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause
# import pytest

import pytest
import pandas as pd
import numpy as np

from mlbox.model.classification.classifier import Classifier
from lightgbm import LGBMClassifier


def test_init_classifier():
    """Test init method of Classifier class."""
    classifier = Classifier()
    assert classifier._Classifier__strategy == "LightGBM"
    assert classifier._Classifier__classif_params == {}
    assert classifier._Classifier__classifier
    assert not classifier._Classifier__col
    assert not classifier._Classifier__fitOK


def test_get_params_classifier():
    """Test get_params method of Classifier class."""
    classifier = Classifier()
    params = classifier.get_params()
    assert params == {'strategy': "LightGBM"}
    assert not classifier._Classifier__classif_params


def test_set_params_classifier():
    """Test set_params method of Classifier class."""
    classifier = Classifier()
    classifier.set_params(strategy="LightGBM")
    assert classifier._Classifier__strategy == "LightGBM"
    classifier.set_params(strategy="RandomForest")
    assert classifier._Classifier__strategy == "RandomForest"
    classifier.set_params(strategy="ExtraTrees")
    assert classifier._Classifier__strategy == "ExtraTrees"
    classifier.set_params(strategy="RandomForest")
    assert classifier._Classifier__strategy == "RandomForest"
    classifier.set_params(strategy="Tree")
    assert classifier._Classifier__strategy == "Tree"
    classifier.set_params(strategy="AdaBoost")
    assert classifier._Classifier__strategy == "AdaBoost"
    classifier.set_params(strategy="Linear")
    assert classifier._Classifier__strategy == "Linear"
    with pytest.warns(UserWarning) as record:
        classifier.set_params(wrong_strategy="wrong_strategy")
    assert len(record) == 1


def test_set_classifier():
    """Test set method of Classifier class."""
    classifier = Classifier()
    with pytest.raises(ValueError):
        classifier._Classifier__set_classifier("wrong_strategy")


def test_fit_classifier():
    """Test fit method of Classifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    classifier = Classifier()
    classifier.fit(df_train, y_train)
    assert np.all(classifier._Classifier__col == df_train.columns)
    assert classifier._Classifier__fitOK


def test_feature_importances_classifier():
    """Test feature_importances method of Classifier class."""
    classifier = Classifier()
    with pytest.raises(ValueError):
        classifier.feature_importances()
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    classifier.set_params(strategy="Linear")
    classifier.fit(df_train, y_train)
    importance = classifier.feature_importances()
    assert importance != {}
    classifier.set_params(strategy="RandomForest")
    classifier.fit(df_train, y_train)
    importance = classifier.feature_importances()
    assert importance != {}
    classifier.set_params(strategy="AdaBoost")
    classifier.fit(df_train, y_train)
    importance = classifier.feature_importances()
    assert importance != {}
    classifier.set_params(strategy="Bagging")
    classifier.fit(df_train, y_train)
    importance = classifier.feature_importances()
    assert importance != {}


def test_predict_classifier():
    """Test predict method of Classifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    classifier = Classifier()
    with pytest.raises(ValueError):
        classifier.predict(df_train)
    classifier.fit(df_train, y_train)
    with pytest.raises(ValueError):
        classifier.predict(None)
    assert len(classifier.predict(df_train)) > 0


def test_predict_log_proba_classifier():
    """Test predict_log_proba method of Classifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    classifier = Classifier(strategy="Linear")
    with pytest.raises(ValueError):
        classifier.predict_log_proba(df_train)
    classifier.fit(df_train, y_train)
    with pytest.raises(ValueError):
        classifier.predict_log_proba(None)
    assert len(classifier.predict_log_proba(df_train)) > 0


def test_predict_proba_classifier():
    """Test predict_proba method of Classifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    classifier = Classifier()
    with pytest.raises(ValueError):
        classifier.predict_proba(df_train)
    classifier.fit(df_train, y_train)
    with pytest.raises(ValueError):
        classifier.predict_proba(None)
    assert len(classifier.predict_proba(df_train)) > 0


def test_score_classifier():
    """Test score method of Classifier class."""
    df_train = pd.read_csv("data_for_tests/clean_train.csv")
    y_train = pd.read_csv("data_for_tests/clean_target.csv", squeeze=True)
    classifier = Classifier()
    with pytest.raises(ValueError):
        classifier.score(df_train, y_train)
    classifier.fit(df_train, y_train)
    with pytest.raises(ValueError):
        classifier.score(None, y_train)
    with pytest.raises(ValueError):
        classifier.score(df_train, None)
    assert classifier.score(df_train, y_train) > 0


def test_get_estimator_classifier():
    """Test get_estimator method of Classifier class."""
    classifier = Classifier()
    estimator = classifier.get_estimator()
    assert isinstance(estimator, type(LGBMClassifier()))
