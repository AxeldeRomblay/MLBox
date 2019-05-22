#!/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
import pytest
import pandas as pd

from mlbox.preprocessing.drift import DriftThreshold


def test_init_drift_threshold():
    drift_threshold = DriftThreshold()
    # assert drift_estimator.n_folds == 2
    # assert drift_estimator.stratify
    # assert drift_estimator.random_state == 1
    # assert not drift_estimator._DriftEstimator__cv
    # assert not drift_estimator._DriftEstimator__pred
    # assert not drift_estimator._DriftEstimator__target
    # assert not drift_estimator._DriftEstimator__fitOK


def test_get_params_drift_threshold():
    drift_threshold = DriftThreshold()
    # dict = {'estimator': drift_threshold.estimator,
    #         'n_folds': 2,
    #         'stratify': True,
    #         'random_state': 1}
    # assert drift_estimator.get_params() == dict

def test_set_params_drift_threshold():
    drift_threshold = DriftThreshold()
    # dict = {'estimator': drift_estimator.estimator,
    #         'n_folds': 3,
    #         'stratify': False,
    #         'random_state': 2}
    # drift_estimator.set_params(**dict)
    # assert drift_estimator.get_params() == dict


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
