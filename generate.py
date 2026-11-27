"""Generates names with the trained model. Run with, for example:

    python generate.py --race dwarf --count 10 --temperature 0.8

This uses the NumPy copy of the model (checkpoints/weights.npz), so it starts
quickly and does not need TensorFlow. Create that file with export_weights.py.
"""

import argparse

import numpy as np

from slm.config import RACES, Config
from slm.name_filter import generate_until_acceptable
from slm.numpy_model import NumpyNameModel
from slm.tokenizer import CharTokenizer


def main() -> None:
    config = Config()

    parser = argparse.ArgumentParser(description="Generate D&D character names.")
    parser.add_argument("--race", choices=RACES, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=config.default_temperature)
    args = parser.parse_args()

    model = NumpyNameModel.load(config.weights_path)
    tokenizer = CharTokenizer.load(config.tokenizer_path)
    rng = np.random.default_rng()

    for _ in range(args.count):
        name = generate_until_acceptable(
            lambda: model.generate_name(
                tokenizer, args.race, args.temperature, rng, config.max_generated_length
            )
        )
        print(name)


if __name__ == "__main__":
    main()
