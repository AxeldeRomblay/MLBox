# coding: utf-8
# Author: Axel ARONIO DE ROMBLAY <axelderomblay@gmail.com>
# License: BSD 3 clause
import pytest

from mlbox.encoding.categorical_encoder import Categorical_encoder


def test_init_encoder():
    encoder = Categorical_encoder()
    assert encoder.strategy == "label_encoding"
    assert encoder.verbose == False
    assert encoder._Categorical_encoder__Lcat == []
    assert encoder._Categorical_encoder__Lnum == []
    assert encoder._Categorical_encoder__Enc == dict()
    assert encoder._Categorical_encoder__K == dict()
    assert encoder._Categorical_encoder__weights == None
    assert encoder._Categorical_encoder__fitOK == False


def test_get_params_encoder():
    encoder = Categorical_encoder()
    dict = {'strategy': "label_encoding",
            'verbose': False}
    assert encoder.get_params() == dict


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
    assert encoder.verbose == True
    encoder.set_params(verbose=False)
    assert encoder.verbose == False
    with pytest.warns(UserWarning) as record:
        encoder.set_params(_Categorical_encoder__Lcat=[])
    assert len(record) == 1
