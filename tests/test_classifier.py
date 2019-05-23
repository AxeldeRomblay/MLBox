#!/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
# import pytest

import pytest
import pandas as pd

from mlbox.model.classification.classifier import Classifier
from mlbox.encoding.categorical_encoder import Categorical_encoder
from lightgbm import LGBMClassifier


def test_init_classifier():
    classifier = Classifier()
    assert classifier._Classifier__strategy == "LightGBM"
    assert classifier._Classifier__classif_params == {}
    assert classifier._Classifier__classifier
    assert not classifier._Classifier__col
    assert not classifier._Classifier__fitOK


def test_get_params_classifier():
    classifier = Classifier()
    params = classifier.get_params()
    assert params == {'strategy': "LightGBM"}
    assert not classifier._Classifier__classif_params


def test_set_params_classifier():
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
    classifier = Classifier()
    with pytest.raises(ValueError):
        classifier._Classifier__set_classifier("wrong_strategy")
