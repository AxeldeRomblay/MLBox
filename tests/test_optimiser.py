# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause
"""Test mlbox.optimisation.optimiser module."""
import pytest
import numpy as np

from mlbox.optimisation.optimiser import Optimiser
from mlbox.preprocessing.drift_thresholder import Drift_thresholder
from mlbox.preprocessing.reader import Reader
from mlbox.optimisation import make_scorer


def test_init_optimiser():
    """Test init method of Optimiser class."""
    with pytest.warns(UserWarning) as record:
        optimiser = Optimiser()
    assert len(record) == 1
    assert not optimiser.scoring
    assert optimiser.n_folds == 2
    assert optimiser.random_state == 1
    assert optimiser.to_path == "save"
    assert optimiser.verbose


def test_get_params_optimiser():
    """Test get_params method of optimiser class."""
    with pytest.warns(UserWarning) as record:
        optimiser = Optimiser()
    assert len(record) == 1
    dict = {'scoring': None,
            'n_folds': 2,
            'random_state': 1,
            'to_path': "save",
            'verbose': True}
    assert optimiser.get_params() == dict


def test_set_params_optimiser():
    """Test set_params method of Optimiser class."""
    with pytest.warns(UserWarning) as record:
        optimiser = Optimiser()
    assert len(record) == 1
    optimiser.set_params(scoring='accuracy')
    assert optimiser.scoring == 'accuracy'
    optimiser.set_params(n_folds=3)
    assert optimiser.n_folds == 3
    optimiser.set_params(random_state=2)
    assert optimiser.random_state == 2
    optimiser.set_params(to_path="name")
    assert optimiser.to_path == "name"
    optimiser.set_params(verbose=False)
    assert not optimiser.verbose
    with pytest.warns(UserWarning) as record:
        optimiser.set_params(wrong_key=3)
    assert len(record) == 1


def test_evaluate_classification_optimiser():
    """Test evaluate method of Optimiser class for classication."""
    reader = Reader(sep=",")
    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv",
                                          "data_for_tests/test.csv"],
                                   target_name="Survived")
    drift_thresholder = Drift_thresholder()
    drift_thresholder = drift_thresholder.fit_transform(dict)

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring=None, n_folds=3)
    assert len(record) == 1
    score = opt.evaluate(None, dict)
    assert -np.Inf <= score

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring="roc_auc", n_folds=3)
    assert len(record) == 1
    score = opt.evaluate(None, dict)
    assert 0. <= score <= 1.

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring="wrong_scoring", n_folds=3)
    assert len(record) == 1
    with pytest.warns(UserWarning) as record:
        score = opt.evaluate(None, dict)
    assert opt.scoring == "neg_log_loss"


def test_evaluate_regression_optimiser():
    """Test evaluate method of Optimiser class for regression."""
    reader = Reader(sep=",")
    dict = reader.train_test_split(Lpath=["data_for_tests/train_regression.csv",
                                          "data_for_tests/test_regression.csv"],
                                   target_name="SalePrice")
    drift_thresholder = Drift_thresholder()
    drift_thresholder = drift_thresholder.fit_transform(dict)

    mape = make_scorer(lambda y_true,
                       y_pred: 100*np.sum(
                                          np.abs(y_true-y_pred)/y_true
                                          )/len(y_true),
                       greater_is_better=False,
                       needs_proba=False)
    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring=mape, n_folds=3)
    assert len(record) == 1
    score = opt.evaluate(None, dict)
    assert -np.Inf <= score

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring=None, n_folds=3)
    assert len(record) == 1
    score = opt.evaluate(None, dict)
    assert -np.Inf <= score

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring="wrong_scoring", n_folds=3)
    assert len(record) == 1
    with pytest.warns(UserWarning) as record:
        score = opt.evaluate(None, dict)
    assert -np.Inf <= score


def test_evaluate_and_optimise_classification():
    """Test evaluate_and_optimise method of Optimiser class."""
    reader = Reader(sep=",")

    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv",
                                          "data_for_tests/test.csv"],
                                   target_name="Survived")
    drift_thresholder = Drift_thresholder()
    drift_thresholder = drift_thresholder.fit_transform(dict)

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring='accuracy', n_folds=3)
    assert len(record) == 1
    dict_error = dict.copy()
    dict_error["target"] = dict_error["target"].astype(str)
    with pytest.raises(ValueError):
        score = opt.evaluate(None, dict_error)

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring='accuracy', n_folds=3)
    assert len(record) == 1
    score = opt.evaluate(None, dict)
    assert 0. <= score <= 1.

    space = {'ne__numerical_strategy': {"search": "choice", "space": [0]},
             'ce__strategy': {"search": "choice",
                              "space": ["label_encoding",
                                        "random_projection",
                                        "entity_embedding"]},
             'fs__threshold': {"search": "uniform",
                               "space": [0.01, 0.3]},
             'est__max_depth': {"search": "choice",
                                "space": [3, 4, 5, 6, 7]}

             }

    best = opt.optimise(space, dict, 1)
    assert type(best) == type(dict)
