# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
import pytest
import pandas as pd

from mlbox.encoding.categorical_encoder import Categorical_encoder


def test_init_encoder():
    encoder = Categorical_encoder()
    assert encoder.strategy == "label_encoding"
    assert not (encoder.verbose)
    assert encoder._Categorical_encoder__Lcat == []
    assert encoder._Categorical_encoder__Lnum == []
    assert encoder._Categorical_encoder__Enc == dict()
    assert encoder._Categorical_encoder__K == dict()
    assert not encoder._Categorical_encoder__weights
    assert not encoder._Categorical_encoder__fitOK


def test_get_params_encoder():
    encoder = Categorical_encoder()
    dict = {'strategy': "label_encoding",
            'verbose': False}
    assert encoder.get_params() == dict


def test_set_params_encoder():
    encoder = Categorical_encoder()
    encoder.set_params(strategy="label_encoding")
    assert encoder.strategy == "label_encoding"
    encoder.set_params(strategy="dummification")
    assert encoder.strategy == "dummification"
    encoder.set_params(strategy="random_projection")
    assert encoder.strategy == "random_projection"
    encoder.set_params(strategy="entity_embedding")
    assert encoder.strategy == "entity_embedding"
    encoder.set_params(verbose=True)
    assert encoder.verbose
    encoder.set_params(verbose=False)
    assert not encoder.verbose
    with pytest.warns(UserWarning) as record:
        encoder.set_params(_Categorical_encoder__Lcat=[])
    assert len(record) == 1


def test_fit_encoder():
    df = pd.read_csv("data_for_tests/train.csv")
    encoder = Categorical_encoder(strategy="wrong_strategy")
    with pytest.raises(ValueError):
        encoder.fit(df, df["Survived"])
    encoder.set_params(strategy="label_encoding")
    encoder.fit(df, df["Survived"])
    assert encoder._Categorical_encoder__fitOK
    encoder.set_params(strategy="dummification")
    encoder.fit(df, df["Survived"])
    assert encoder._Categorical_encoder__fitOK
    encoder.set_params(strategy="random_projection")
    encoder.fit(df, df["Survived"])
    assert encoder._Categorical_encoder__fitOK
    encoder.set_params(strategy="entity_embedding")
    encoder.fit(df, df["Survived"])
    assert encoder._Categorical_encoder__fitOK

def test_transform_encoder():
    df = pd.read_csv("data_for_tests/train.csv")
    encoder = Categorical_encoder()
    with pytest.raises(ValueError):
        encoder.transform(df)
    encoder.fit(df, df["Survived"])
    df_encoded = encoder.transform(df)
    assert (df.columns == df_encoded.columns).all()
