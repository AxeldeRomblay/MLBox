#!/usr/bin/env python
# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
import pytest
import pandas as pd

from mlbox.encoding.na_encoder import NA_encoder


def test_init_NA_encoder():
    encoder = NA_encoder()
    assert encoder.numerical_strategy == "mean"
    assert encoder.categorical_strategy == "<NULL>"
    assert encoder._NA_encoder__Lcat == []
    assert encoder._NA_encoder__Lnum == []
    assert not encoder._NA_encoder__imp
    assert encoder._NA_encoder__mode == dict()
    assert not encoder._NA_encoder__fitOK


def test_get_params_NA_encoder():
    encoder = NA_encoder()
    dict = {'numerical_strategy': "mean",
            'categorical_strategy': "<NULL>"}
    assert encoder.get_params() == dict


def test_set_params_NA_encoder():
    encoder = NA_encoder()

    encoder.set_params(numerical_strategy="mean")
    assert encoder.numerical_strategy == "mean"
    encoder.set_params(numerical_strategy="median")
    assert encoder.numerical_strategy == "median"
    encoder.set_params(numerical_strategy="most_frequent")
    assert encoder.numerical_strategy == "most_frequent"
    encoder.set_params(numerical_strategy=3.0)
    assert encoder.numerical_strategy == 3.0

    encoder.set_params(categorical_strategy="<NULL>")
    assert encoder.categorical_strategy == "<NULL>"
    encoder.set_params(categorical_strategy="most_frequent")
    assert encoder.categorical_strategy == "most_frequent"
    encoder.set_params(categorical_strategy="string_test")
    assert encoder.categorical_strategy == "string_test"

    with pytest.warns(UserWarning) as record:
        encoder.set_params(_Categorical_encoder__Lcat=[])
    assert len(record) == 1


def test_fit_NA_encoder():
    df = pd.read_csv("data_for_tests/train.csv")

    encoder = NA_encoder(numerical_strategy="wrong_strategy")
    with pytest.raises(ValueError):
        encoder.fit(df, df["Survived"])
    encoder.set_params(numerical_strategy="mean")
    encoder.fit(df, df["Survived"])
    assert encoder._NA_encoder__fitOK
    encoder.set_params(numerical_strategy="median")
    encoder.fit(df, df["Survived"])
    assert encoder._NA_encoder__fitOK
    encoder.set_params(numerical_strategy="most_frequent")
    encoder.fit(df, df["Survived"])
    assert encoder._NA_encoder__fitOK
    encoder.set_params(numerical_strategy=3.0)
    encoder.fit(df, df["Survived"])
    assert encoder._NA_encoder__fitOK

    encoder = NA_encoder(categorical_strategy=2)
    with pytest.raises(ValueError):
        encoder.fit(df, df["Survived"])
    encoder.set_params(categorical_strategy="<NULL>")
    encoder.fit(df, df["Survived"])
    assert encoder._NA_encoder__fitOK
    encoder.set_params(categorical_strategy="most_frequent")
    encoder.fit(df, df["Survived"])


def test_transform_NA_encoder():
    df = pd.read_csv("data_for_tests/train.csv")
    encoder = NA_encoder()
    with pytest.raises(ValueError):
        encoder.transform(df)
    encoder.fit(df, df["Survived"])
    df_encoded = encoder.transform(df)
    assert (df.columns == df_encoded.columns).all()
