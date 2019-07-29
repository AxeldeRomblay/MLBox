# !/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# Author: Henri GERARD <hgerard.pro@gmail.com>
# License: BSD 3 clause
"""Test mlbox.prediction0.predictor module."""
import sys
import pytest
import numpy as np
import pandas as pd

from mlbox.prediction.predictor import Predictor
from mlbox.optimisation.optimiser import Optimiser
from mlbox.preprocessing.drift_thresholder import Drift_thresholder
from mlbox.preprocessing.reader import Reader
from mlbox.optimisation import make_scorer

if sys.version_info[0] >= 3:
    from unittest.mock import patch


set_backend = "import matplotlib\nmatplotlib.use('Agg')\n"


def test_init_predictor():
    """Test init method of Predictor class."""
    predictor = Predictor()
    assert predictor.to_path == "save"
    assert predictor.verbose


def test_get_params_predictor():
    """Test get_params method of Predictor class."""
    predictor = Predictor()
    dict = {'to_path': "save",
            'verbose': True}
    assert predictor.get_params() == dict


def test_set_params_predictor():
    """Test set_params method of Predictor class."""
    predictor = Predictor()
    predictor.set_params(to_path="name")
    assert predictor.to_path == "name"
    predictor.set_params(verbose=False)
    assert not predictor.verbose
    with pytest.warns(UserWarning) as record:
        predictor.set_params(wrong_key=3)
    assert len(record) == 1


def test_fit_predict_predictor_classification():
    """Test fit_predict method of Predictor class for classification."""
    reader = Reader(sep=",")
    dict = reader.train_test_split(Lpath=["data_for_tests/train.csv",
                                          "data_for_tests/test.csv"],
                                   target_name="Survived")
    drift_thresholder = Drift_thresholder()
    drift_thresholder = drift_thresholder.fit_transform(dict)

    with pytest.warns(UserWarning) as record:
        opt = Optimiser(scoring='accuracy', n_folds=3)
    assert len(record) == 1

    space = {'ne__numerical_strategy': {"search": "choice", "space": [0]},
             'ce__strategy': {"search": "choice",
                              "space": ["entity_embedding"]},
             'fs__threshold': {"search": "uniform",
                               "space": [0.01, 0.3]},
             'est__max_depth': {"search": "choice",
                                "space": [3, 4, 5, 6, 7]}

             }

    optimal_hyper_parameters = opt.optimise(space, dict, 1)

    predictor = Predictor(verbose=False)
    predictor.fit_predict(optimal_hyper_parameters, dict)
    pred_df = pd.read_csv("save/Survived_predictions.csv")
    assert np.all(list(pred_df.columns) == ['Unnamed: 0',
                                            '0.0',
                                            '1.0',
                                            'Survived_predicted'])
    assert np.shape(pred_df) == (418, 4)


if sys.version_info[0] >= 3:
    @patch('matplotlib.pyplot.show')
    def test_fit_predict_predictor_regression(mock_show):
        """Test fit_predict method of Predictor class for regression."""
        rd = Reader(sep=',')
        dt = rd.train_test_split(Lpath=["data_for_tests/train_regression.csv",
                                        "data_for_tests/test_regression.csv"],
                                 target_name="SalePrice")

        drift_thresholder = Drift_thresholder()
        df = drift_thresholder.fit_transform(dt)

        mape = make_scorer(lambda y_true,
                           y_pred: 100*np.sum(
                                              np.abs(y_true-y_pred)/y_true
                                              )/len(y_true),
                           greater_is_better=False,
                           needs_proba=False)
        opt = Optimiser(scoring=mape, n_folds=3)

        opt.evaluate(None, df)

        space = {
                'ne__numerical_strategy': {"search": "choice",
                                           "space": [0]},
                'ce__strategy': {"search": "choice",
                                 "space": ["label_encoding",
                                           "random_projection",
                                           "entity_embedding"]},
                'fs__threshold': {"search": "uniform",
                                  "space": [0.01, 0.3]},
                'est__max_depth': {"search": "choice",
                                   "space": [3, 4, 5, 6, 7]}

                }

        best = opt.optimise(space, df, 1)

        prd = Predictor(verbose=True)
        prd.fit_predict(best, df)
        pred_df = pd.read_csv("save/SalePrice_predictions.csv")
        assert np.all(list(pred_df.columns) == ['Unnamed: 0',
                                                'SalePrice_predicted'])
        assert np.shape(pred_df) == (1459, 2)

else:
    def test_fit_predict_predictor_regression():
        """Test fit_predict method of Predictor class for regression."""
        rd = Reader(sep=',')
        dt = rd.train_test_split(Lpath=["data_for_tests/train_regression.csv",
                                        "data_for_tests/test_regression.csv"],
                                 target_name="SalePrice")

        drift_thresholder = Drift_thresholder()
        df = drift_thresholder.fit_transform(dt)

        mape = make_scorer(lambda y_true,
                           y_pred: 100*np.sum(
                                              np.abs(y_true-y_pred)/y_true
                                              )/len(y_true),
                           greater_is_better=False,
                           needs_proba=False)
        opt = Optimiser(scoring=mape, n_folds=3)

        opt.evaluate(None, df)

        space = {
                'ne__numerical_strategy': {"search": "choice",
                                           "space": [0]},
                'ce__strategy': {"search": "choice",
                                 "space": ["label_encoding",
                                           "random_projection",
                                           "entity_embedding"]},
                'fs__threshold': {"search": "uniform",
                                  "space": [0.01, 0.3]},
                'est__max_depth': {"search": "choice",
                                   "space": [3, 4, 5, 6, 7]}

                }

        best = opt.optimise(space, df, 1)

        prd = Predictor(verbose=False)
        prd.fit_predict(best, df)
        pred_df = pd.read_csv("save/SalePrice_predictions.csv")
        assert np.all(list(pred_df.columns) == ['Unnamed: 0',
                                                'SalePrice_predicted'])
        assert np.shape(pred_df) == (1459, 2)
