"""Decides whether a generated name may be shown to the user.

The model samples one character at a time, so now and then it produces names no
person would write, such as "Filllich", or names of real people that must never
appear in a fantasy name generator. Rejected names are simply generated again.
"""

import re
from collections.abc import Callable

# Real people whose names must never be generated. Written in lowercase, because
# generate_name capitalises the first letter of every name. The model has no space
# character, so a two-word name is listed without a space and with a hyphen.
BANNED_NAMES = frozenset(
    {
        # Dictators
        "hitler",
        "stalin",
        "mussolini",
        "mao",
        "polpot",
        "pol-pot",
        "franco",
        "pinochet",
        "gaddafi",
        "saddam",
        "ceausescu",
        # Leading Nazis
        "himmler",
        "goebbels",
        "eichmann",
        "mengele",
    }
)

# No training name has three identical letters in a row, so the model should not either.
MAX_REPEATED_LETTERS = 2


def has_too_many_repeated_letters(name: str) -> bool:
    """Returns True if one character appears more than MAX_REPEATED_LETTERS times in a row."""
    # (.) remembers any character, \1{2} asks for that same character twice more.
    pattern = rf"(.)\1{{{MAX_REPEATED_LETTERS}}}"
    return re.search(pattern, name) is not None


def is_acceptable(name: str) -> bool:
    """Returns False if the name repeats a letter too often or is a banned name."""
    lowered = name.lower()
    if lowered in BANNED_NAMES:
        return False
    return not has_too_many_repeated_letters(lowered)


def generate_until_acceptable(make_name: Callable[[], str], max_attempts: int = 100) -> str:
    """Calls make_name until it returns an acceptable name.

    make_name is any function without arguments that returns a new name, so this
    works with every model. The limit stops an endless loop, for example at a very
    high temperature where almost every name is nonsense.
    """
    for _ in range(max_attempts):
        name = make_name()
        if is_acceptable(name):
            return name
    raise RuntimeError(f"No acceptable name after {max_attempts} attempts.")
