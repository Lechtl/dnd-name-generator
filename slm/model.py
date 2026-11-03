"""The neural network: a character-level GRU whose memory starts from the race.

Data flow for one name:

    race  -> race embedding -> Dense -> the GRU's starting memory
    chars -> char embedding -> GRU   -> Dense -> a score for every possible next character
"""

import keras
from keras import layers

from slm.config import Config


def build_model(vocab_size: int, num_races: int, config: Config) -> keras.Model:
    # None means names can have any length.
    chars = keras.Input(shape=(None,), dtype="int32", name="chars")
    race = keras.Input(shape=(), dtype="int32", name="race")

    # Embeddings give each character and race a learned vector instead of a bare ID.
    # IDs have no meaningful order, while vectors let similar items end up close
    # together, such as vowels, or races with similar-sounding names.
    char_vectors = layers.Embedding(vocab_size, config.char_embedding_dim, name="char_embedding")(chars)
    race_vector = layers.Embedding(num_races, config.race_embedding_dim, name="race_embedding")(race)

    # The GRU's memory is a vector of size gru_units, so this layer converts the race
    # into that size. Each race now starts a name with a different memory, which is
    # how the same model can write in nine different styles. tanh is used because it
    # produces the same -1 to 1 range the GRU's memory uses.
    initial_state = layers.Dense(config.gru_units, activation="tanh", name="race_to_state")(race_vector)

    # The GRU reads one character at a time and updates its memory after each one.
    # return_sequences=True gives an output for every position, because we want a
    # next-character prediction at every position, not only at the end of the name.
    sequence = layers.GRU(config.gru_units, return_sequences=True, name="gru")(
        char_vectors, initial_state=[initial_state]
    )

    # Dropout randomly switches off some values during training, which makes
    # memorising individual names harder.
    sequence = layers.Dropout(config.dropout, name="dropout")(sequence)

    # One raw score (a "logit") per vocabulary entry at each position. The softmax
    # that turns these into probabilities is applied in the loss and during sampling.
    logits = layers.Dense(vocab_size, name="next_char_logits")(sequence)

    return keras.Model(inputs={"chars": chars, "race": race}, outputs=logits, name="name_generator")
