"""Saves the trained Keras model's weights as plain NumPy arrays.

Run this once after training:

    python export_weights.py

Afterwards generate.py and the web app only need NumPy, not TensorFlow.
"""

from pathlib import Path

import keras
import numpy as np

from slm.config import Config


def extract_weights(model: keras.Model) -> dict[str, np.ndarray]:
    """Returns every weight under a name such as "gru_kernel" or "char_embedding"."""
    weights = {}
    for layer in model.layers:
        for weight in layer.weights:
            # Keras names weights like "gru/gru_cell/kernel"; keep the layer name and
            # the weight's own name. Embeddings have only one weight, so their layer
            # name is enough.
            layer_name, weight_name = weight.path.split("/")[0], weight.path.split("/")[-1]
            key = layer_name if weight_name == "embeddings" else f"{layer_name}_{weight_name}"
            weights[key] = weight.numpy()
    return weights


def export_weights(model_path: Path, weights_path: Path) -> None:
    model = keras.models.load_model(model_path, compile=False)
    np.savez(weights_path, **extract_weights(model))


if __name__ == "__main__":
    config = Config()
    export_weights(config.model_path, config.weights_path)
    print(f"Saved weights to {config.weights_path}")
