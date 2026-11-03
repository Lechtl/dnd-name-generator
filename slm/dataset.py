"""Loads the raw names and turns them into batches the model can train on.

Expected input is a CSV file with the columns `name` and `race`, for example:

    name,race
    Thorin,dwarf
    Legolas,elf
"""

import csv
import random
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from slm.config import RACES
from slm.tokenizer import PAD_ID, CharTokenizer


def load_names(path: Path) -> list[tuple[str, str]]:
    """Reads the CSV and returns unique, cleaned (name, race) pairs."""
    samples = set()
    # utf-8-sig also handles the hidden marker Excel adds to the start of CSV files.
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            name = row["name"].strip().lower()
            race = row["race"].strip().lower()
            if race not in RACES:
                raise ValueError(f"Unknown race {race!r} for name {name!r}")
            if name:
                samples.add((name, race))
    # Sets have no fixed order, so sort to keep runs reproducible.
    return sorted(samples)


def split_by_name(
    samples: list[tuple[str, str]], validation_fraction: float, seed: int
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Splits the samples into training and validation sets.

    The split is made over distinct names, not over rows. If a name is listed for
    two races, both rows land on the same side. Otherwise the model would have seen
    validation names during training and the validation loss would look too good.
    """
    names = sorted({name for name, _ in samples})
    random.Random(seed).shuffle(names)
    validation_names = set(names[: int(len(names) * validation_fraction)])

    train = [s for s in samples if s[0] not in validation_names]
    validation = [s for s in samples if s[0] in validation_names]
    return train, validation


def build_dataset(
    samples: list[tuple[str, str]],
    tokenizer: CharTokenizer,
    batch_size: int,
    shuffle: bool = False,
) -> tf.data.Dataset:
    """Creates the (inputs, targets) batches for training.

    The target is the input shifted by one position, so at every position the model
    learns to predict the next character. For "bo":

        input:   START  b  o
        target:  b      o  END
    """
    encoded = [tokenizer.encode(name) for name, _ in samples]
    # Padding goes at the end, so it comes after everything the model needs to read.
    padded = keras.utils.pad_sequences(encoded, padding="post", value=PAD_ID)

    inputs = {
        "chars": padded[:, :-1],
        "race": np.array([RACES.index(race) for _, race in samples]),
    }
    targets = padded[:, 1:]

    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(len(samples))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
