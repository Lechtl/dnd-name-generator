"""Checks that the NumPy model computes exactly what the trained Keras model computes."""

import keras
import numpy as np
import pytest

from export_weights import extract_weights
from slm.config import RACES, Config
from slm.numpy_model import NumpyNameModel
from slm.tokenizer import START_ID, CharTokenizer


@pytest.fixture(scope="module")
def models():
    # Loading the Keras model is slow, so all tests in this file share one copy.
    config = Config()
    keras_model = keras.models.load_model(config.model_path, compile=False)
    numpy_model = NumpyNameModel(extract_weights(keras_model))
    tokenizer = CharTokenizer.load(config.tokenizer_path)
    return keras_model, numpy_model, tokenizer


@pytest.mark.parametrize("race", ["dwarf", "elf", "tiefling"])
def test_numpy_logits_match_keras(models, race):
    keras_model, numpy_model, tokenizer = models
    # START followed by the letters of "thorin", i.e. what the model reads.
    ids = [START_ID] + [tokenizer.char_to_id[c] for c in "thorin"]

    keras_logits = np.array(
        keras_model(
            {"chars": np.array([ids]), "race": np.array([RACES.index(race)])}, training=False
        )
    )[0]

    h = numpy_model.initial_state(RACES.index(race))
    numpy_logits = []
    for char_id in ids:
        logits, h = numpy_model.step(char_id, h)
        numpy_logits.append(logits)

    # Both compute in float32, so only tiny rounding differences are allowed.
    np.testing.assert_allclose(np.array(numpy_logits), keras_logits, atol=1e-4)


def test_exported_file_loads_the_same_weights(models, tmp_path):
    keras_model, numpy_model, _ = models
    path = tmp_path / "weights.npz"
    np.savez(path, **extract_weights(keras_model))

    loaded = NumpyNameModel.load(path)

    np.testing.assert_array_equal(loaded.gru_kernel, numpy_model.gru_kernel)
    np.testing.assert_array_equal(loaded.output_bias, numpy_model.output_bias)
