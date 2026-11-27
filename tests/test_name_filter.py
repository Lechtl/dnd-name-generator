"""Tests for the filter that rejects unwanted generated names."""

import pytest

from slm.config import Config
from slm.name_filter import BANNED_NAMES, generate_until_acceptable, is_acceptable
from slm.tokenizer import CharTokenizer


@pytest.mark.parametrize("name", ["Thorin", "Galadriel", "Anna", "Elladan", "A"])
def test_accepts_normal_names(name):
    # Double letters are fine: 372 training names have one.
    assert is_acceptable(name)


@pytest.mark.parametrize("name", ["Filllich", "Aaab", "Thorinnn", "Ooo"])
def test_rejects_three_identical_letters_in_a_row(name):
    # Covers the start, the middle and the end, and a capital mixed with lowercase.
    assert not is_acceptable(name)


def test_accepts_the_same_letter_three_times_when_not_in_a_row():
    assert is_acceptable("Lalal")


@pytest.mark.parametrize("name", ["Hitler", "hitler", "HITLER", "Stalin", "Mao", "Pol-pot"])
def test_rejects_banned_names_in_any_case(name):
    assert not is_acceptable(name)


def test_banned_list_is_lowercase():
    # The filter compares lowercase names, so a capital in the list would never match.
    assert all(name == name.lower() for name in BANNED_NAMES)


def test_banned_names_only_use_characters_the_model_knows():
    # An entry like "pol pot" could never be generated, because the vocabulary has
    # no space, so it would silently block nothing.
    tokenizer = CharTokenizer.load(Config().tokenizer_path)

    for name in BANNED_NAMES:
        assert set(name) <= set(tokenizer.chars), name


def test_generate_until_acceptable_skips_rejected_names():
    # A stand-in for the model that returns these names one after another.
    names = iter(["Hitler", "Filllich", "Thorin"])

    assert generate_until_acceptable(lambda: next(names)) == "Thorin"


def test_generate_until_acceptable_gives_up_after_max_attempts():
    with pytest.raises(RuntimeError):
        generate_until_acceptable(lambda: "Hitler", max_attempts=5)
