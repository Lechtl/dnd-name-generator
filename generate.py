"""Generates names with a trained model. Run with, for example:

    python generate.py --race dwarf --count 10 --temperature 0.8
"""

import argparse

import keras
import numpy as np

from slm.config import RACES, Config
from slm.tokenizer import END_ID, PAD_ID, START_ID, CharTokenizer


def sample_next_id(logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Picks the next character ID at random, weighted by the model's scores.

    Temperature controls the risk taken. Below 1 favours likely characters, giving
    safer and more conventional names. Above 1 flattens the odds, giving more
    unusual names and more nonsense.
    """
    # float64 avoids rounding errors that make rng.choice reject the probabilities.
    logits = logits.astype(np.float64) / temperature

    # These two can never be a sensible next character, so rule them out.
    logits[[PAD_ID, START_ID]] = -np.inf

    # This is the softmax function. Subtracting the maximum first keeps exp() from
    # overflowing, and does not change the resulting probabilities.
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


def generate_name(
    model: keras.Model,
    tokenizer: CharTokenizer,
    race: str,
    config: Config,
    temperature: float,
    rng: np.random.Generator,
) -> str:
    """Builds a name one character at a time, feeding each choice back into the model."""
    race_id = np.array([RACES.index(race)])
    ids = [START_ID]

    for _ in range(config.max_generated_length):
        # Re-reads the whole name so far at every step. This repeats some work, but
        # names are so short that the simpler code is well worth it.
        logits = model({"chars": np.array([ids]), "race": race_id}, training=False)
        # Shape is (batch, position, vocabulary): first name, latest position.
        next_id = sample_next_id(np.array(logits[0, -1]), temperature, rng)
        if next_id == END_ID:
            break
        ids.append(next_id)

    name = tokenizer.decode(ids)
    # Names were lowercased for training, so restore the leading capital.
    return name[:1].upper() + name[1:]


def main() -> None:
    config = Config()

    parser = argparse.ArgumentParser(description="Generate D&D character names.")
    parser.add_argument("--race", choices=RACES, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=config.default_temperature)
    args = parser.parse_args()

    # compile=False skips restoring the optimizer and loss, which only training needs.
    model = keras.models.load_model(config.model_path, compile=False)
    tokenizer = CharTokenizer.load(config.tokenizer_path)
    rng = np.random.default_rng()

    for _ in range(args.count):
        print(generate_name(model, tokenizer, args.race, config, args.temperature, rng))


if __name__ == "__main__":
    main()
