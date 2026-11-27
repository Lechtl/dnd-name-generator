"""Tests for the character tokenizer. Run all tests with: python -m pytest"""

import pytest

from slm.tokenizer import END_ID, NUM_SPECIAL_TOKENS, PAD_ID, START_ID, CharTokenizer


def test_special_tokens_have_fixed_ids():
    # The saved model depends on these exact numbers, so a change must fail loudly.
    assert (PAD_ID, START_ID, END_ID) == (0, 1, 2)


def test_fit_builds_sorted_vocabulary_after_special_tokens():
    tokenizer = CharTokenizer.fit(["ana", "anna", "ada"])

    assert tokenizer.chars == ["a", "d", "n"]
    assert tokenizer.char_to_id == {"a": 3, "d": 4, "n": 5}
    assert tokenizer.vocab_size == 3 + NUM_SPECIAL_TOKENS


def test_fit_does_not_depend_on_input_order():
    first = CharTokenizer.fit(["bo", "ada"])
    second = CharTokenizer.fit(["ada", "bo"])

    assert first.chars == second.chars


def test_encode_wraps_name_in_start_and_end():
    tokenizer = CharTokenizer.fit(["ana", "anna", "ada"])

    assert tokenizer.encode("ana") == [START_ID, 3, 5, 3, END_ID]


def test_decode_skips_special_tokens():
    tokenizer = CharTokenizer.fit(["ana", "anna", "ada"])

    assert tokenizer.decode([START_ID, 4, 3, END_ID, PAD_ID, PAD_ID]) == "da"


@pytest.mark.parametrize("name", ["ana", "anna", "ada", "d", "nandana"])
def test_decode_reverses_encode(name):
    tokenizer = CharTokenizer.fit(["ana", "anna", "ada"])

    assert tokenizer.decode(tokenizer.encode(name)) == name


def test_encode_rejects_unknown_character():
    tokenizer = CharTokenizer.fit(["ana"])

    # Capital letters never reach the tokenizer in training, because
    # load_names lowercases every name first.
    with pytest.raises(KeyError):
        tokenizer.encode("Ana")


def test_save_and_load_keep_the_same_ids(tmp_path):
    original = CharTokenizer.fit(["thorin", "legolas", "o'neil", "half-elf"])
    path = tmp_path / "tokenizer.json"

    original.save(path)
    loaded = CharTokenizer.load(path)

    assert loaded.char_to_id == original.char_to_id
