"""Tests for loading, splitting and batching the names."""

import numpy as np
import pytest

from slm.config import RACES
from slm.dataset import build_dataset, load_names, split_by_name
from slm.tokenizer import END_ID, PAD_ID, START_ID, CharTokenizer


def write_csv(tmp_path, text):
    path = tmp_path / "names.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_names_cleans_and_deduplicates(tmp_path):
    path = write_csv(
        tmp_path,
        "name,surname,race,gender\n"
        " Thorin ,Oakenshield,Dwarf,male\n"
        "thorin,,dwarf,male\n"
        ",Baggins,halfling,any\n"
        "Legolas,,elf,male\n",
    )

    # Spaces and capitals are removed, the duplicate Thorin is kept once,
    # and the surname-only row is skipped.
    assert load_names(path) == [("legolas", "elf"), ("thorin", "dwarf")]


def test_load_names_rejects_unknown_race(tmp_path):
    path = write_csv(tmp_path, "name,race\nGandalf,wizard\n")

    with pytest.raises(ValueError, match="wizard"):
        load_names(path)


def test_split_keeps_every_sample_exactly_once():
    samples = [(f"name{i}", RACES[i % len(RACES)]) for i in range(100)]

    train, validation = split_by_name(samples, validation_fraction=0.1, seed=42)

    assert sorted(train + validation) == sorted(samples)
    assert len(validation) == 10


def test_split_is_reproducible_with_the_same_seed():
    samples = [(f"name{i}", "elf") for i in range(50)]

    assert split_by_name(samples, 0.2, seed=1) == split_by_name(samples, 0.2, seed=1)


def test_split_puts_a_name_on_only_one_side():
    # Every name is listed under two races, like denethor (elf and human) in the
    # real data. Only then could a bad split put the same name on both sides.
    samples = [(f"name{i}", race) for i in range(20) for race in ("elf", "human")]

    train, validation = split_by_name(samples, validation_fraction=0.2, seed=42)

    train_names = {name for name, _ in train}
    validation_names = {name for name, _ in validation}
    # An empty validation set would make the next check pass without testing anything.
    assert validation_names
    # & keeps the names that are in both sets. There must be none.
    assert train_names & validation_names == set()


def test_targets_are_inputs_shifted_by_one():
    samples = [("ana", "elf"), ("anna", "dwarf")]
    tokenizer = CharTokenizer.fit([name for name, _ in samples])

    inputs, targets = next(iter(build_dataset(samples, tokenizer, batch_size=2)))
    chars, targets = inputs["chars"].numpy(), targets.numpy()

    # "anna" is the longest: START a n n a END, so 5 input and 5 target positions.
    assert chars[1].tolist() == [START_ID, 3, 4, 4, 3]
    assert targets[1].tolist() == [3, 4, 4, 3, END_ID]
    # "ana" is one shorter, so its last target is padding.
    assert targets[0].tolist() == [3, 4, 3, END_ID, PAD_ID]
    # Every target is the next input.
    np.testing.assert_array_equal(targets[:, :-1], chars[:, 1:])
    assert inputs["race"].numpy().tolist() == [RACES.index("elf"), RACES.index("dwarf")]
