"""Runs the trained model with NumPy only, so generating names does not need TensorFlow.

TensorFlow takes several seconds to import and adds overhead to every model call.
For generation that is wasteful: the model is small, and each step is only a few
matrix multiplications. This module repeats exactly what the Keras model computes,
using the weights exported by export_weights.py. Training still uses Keras.

Unlike the Keras version, this reads one character per step and carries the GRU's
memory forward, instead of re-reading the whole name at every step.
"""

from pathlib import Path

import numpy as np

from slm.config import RACES
from slm.sampling import sample_next_id
from slm.tokenizer import END_ID, START_ID, CharTokenizer


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class NumpyNameModel:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.char_embedding = weights["char_embedding"]  # (vocab, 32)
        self.race_embedding = weights["race_embedding"]  # (races, 16)
        self.race_kernel = weights["race_to_state_kernel"]  # (16, 128)
        self.race_bias = weights["race_to_state_bias"]  # (128,)
        # Keras stores the three GRU parts side by side, in the order
        # update gate z, reset gate r, candidate. Each part is `units` columns wide.
        self.gru_kernel = weights["gru_kernel"]  # (32, 3 * 128), applied to the character
        self.gru_recurrent_kernel = weights["gru_recurrent_kernel"]  # (128, 3 * 128), applied to the memory
        self.gru_input_bias = weights["gru_bias"][0]  # (3 * 128,)
        self.gru_recurrent_bias = weights["gru_bias"][1]  # (3 * 128,)
        self.output_kernel = weights["next_char_logits_kernel"]  # (128, vocab)
        self.output_bias = weights["next_char_logits_bias"]  # (vocab,)
        self.units = self.race_bias.shape[0]

    @classmethod
    def load(cls, path: Path) -> "NumpyNameModel":
        with np.load(path) as file:
            return cls({key: file[key] for key in file.files})

    def initial_state(self, race_id: int) -> np.ndarray:
        """The GRU's starting memory for a race: race embedding, then Dense with tanh."""
        race_vector = self.race_embedding[race_id]
        return np.tanh(race_vector @ self.race_kernel + self.race_bias)

    def gru_step(self, x: np.ndarray, h: np.ndarray) -> np.ndarray:
        """One GRU step: reads the character vector x and returns the new memory.

        x has shape (32,) and h has shape (128,). Keras uses the variant with
        reset_after=True, where the reset gate is applied after the memory has been
        multiplied by the recurrent kernel.
        """
        u = self.units
        # Both products at once, then split into the three parts z, r and candidate.
        x_parts = x @ self.gru_kernel + self.gru_input_bias
        h_parts = h @ self.gru_recurrent_kernel + self.gru_recurrent_bias
        x_z, x_r, x_c = x_parts[:u], x_parts[u : 2 * u], x_parts[2 * u :]
        h_z, h_r, h_c = h_parts[:u], h_parts[u : 2 * u], h_parts[2 * u :]

        z = sigmoid(x_z + h_z)  # update gate: how much of the old memory to keep
        r = sigmoid(x_r + h_r)  # reset gate: how much of the old memory feeds the candidate
        # reset_after=True: r scales the memory's part after the multiplication.
        c = np.tanh(x_c + r * h_c)  # candidate: the proposed new memory
        # Keras convention: z = 1 keeps the old memory, z = 0 takes the candidate.
        # Some textbooks swap the roles of z and 1 - z; the result is the same idea.
        return z * h + (1.0 - z) * c

    def step(self, char_id: int, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reads one character and returns the scores for the next one and the new memory."""
        h = self.gru_step(self.char_embedding[char_id], h)
        # Dropout is only active during training, so it is simply left out here.
        logits = h @ self.output_kernel + self.output_bias
        return logits, h

    def generate_name(
        self,
        tokenizer: CharTokenizer,
        race: str,
        temperature: float,
        rng: np.random.Generator,
        max_length: int,
    ) -> str:
        """Builds a name one character at a time, feeding each choice back in."""
        h = self.initial_state(RACES.index(race))
        ids = [START_ID]

        for _ in range(max_length):
            logits, h = self.step(ids[-1], h)
            next_id = sample_next_id(logits, temperature, rng)
            if next_id == END_ID:
                break
            ids.append(next_id)

        name = tokenizer.decode(ids)
        # Names were lowercased for training, so restore the leading capital.
        return name[:1].upper() + name[1:]
