#!/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
import pytest
import pandas as pd

from mlbox.prediction.predictor import Predictor
from mlbox.optimisation.optimiser import Optimiser
from mlbox.preprocessing.drift_thresholder import Drift_thresholder
from mlbox.preprocessing.reader import Reader


def test_init_predictor():
    predictor = Predictor()
    assert predictor.to_path == "save"
    assert predictor.verbose


def test_get_params_predictor():
    predictor = Predictor()
    dict = {'to_path': "save",
            'verbose': True}
    assert predictor.get_params() == dict


def test_set_params_predictor():
    predictor = Predictor()
    predictor.set_params(to_path="name")
    assert predictor.to_path == "name"
    predictor.set_params(verbose=False)
    assert not predictor.verbose
    with pytest.warns(UserWarning) as record:
        predictor.set_params(wrong_key=3)
    assert len(record) == 1


def test_fit_predict():
    reader = Reader(sep=",")
    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv",
                                          "data_for_tests/test.csv"],
                                   target_name="Survived")
    drift_thresholder = Drift_thresholder()
    drift_thresholder = drift_thresholder.fit_transform(dict)

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring='accuracy', n_folds=3)
    assert len(record) == 1
    score = opt.evaluate(None, dict)

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

    optimal_hyper_parameters = opt.optimise(space, dict, 1)

    predictor = Predictor()
    predictor.fit_predict(optimal_hyper_parameters, dict)
