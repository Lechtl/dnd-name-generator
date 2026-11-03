"""Converts names to lists of integers and back, one character at a time.

Neural networks only work with numbers, so every character gets an ID. We tokenize
by character rather than by word because a name is a single word: the patterns we
want to learn are the letter combinations inside it.
"""

import json
from pathlib import Path

# Special tokens that never appear in a name but tell the model about its structure.
PAD_ID = 0  # Fills unused positions, so all names in a batch have the same length.
START_ID = 1  # The model's first input, meaning "a new name begins here".
END_ID = 2  # The model predicts this when it thinks the name is finished.
NUM_SPECIAL_TOKENS = 3


class CharTokenizer:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.char_to_id = {c: i + NUM_SPECIAL_TOKENS for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}

    @classmethod
    def fit(cls, names: list[str]) -> "CharTokenizer":
        """Builds the vocabulary from every distinct character in the names."""
        # Sorting makes the IDs the same on every run.
        return cls(sorted(set("".join(names))))

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + NUM_SPECIAL_TOKENS

    def encode(self, name: str) -> list[int]:
        """Example: "bo" -> [START_ID, id of b, id of o, END_ID]."""
        return [START_ID] + [self.char_to_id[c] for c in name] + [END_ID]

    def decode(self, ids: list[int]) -> str:
        """Turns IDs back into a name, skipping the special tokens."""
        return "".join(self.id_to_char[i] for i in ids if i >= NUM_SPECIAL_TOKENS)

    def save(self, path: Path) -> None:
        # The vocabulary is saved next to the model, because the model's output
        # only makes sense with the exact character-to-ID mapping it trained on.
        path.write_text(json.dumps({"chars": self.chars}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8"))["chars"])
