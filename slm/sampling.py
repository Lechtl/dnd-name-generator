"""Picks the next character from the model's scores.

Kept free of TensorFlow, so both the Keras model and the NumPy model can use it.
"""

import numpy as np

from slm.tokenizer import PAD_ID, START_ID


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
