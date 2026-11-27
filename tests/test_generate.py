"""Tests for picking the next character during generation."""

import numpy as np

from generate import sample_next_id
from slm.tokenizer import END_ID, PAD_ID, START_ID


def test_never_samples_pad_or_start():
    # PAD and START get by far the highest scores, but must still never be picked.
    logits = np.array([10.0, 10.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(0)

    picks = {sample_next_id(logits, temperature=1.0, rng=rng) for _ in range(200)}

    assert PAD_ID not in picks
    assert START_ID not in picks


def test_low_temperature_picks_the_favourite():
    logits = np.array([0.0, 0.0, 2.0, 1.0, 3.0])
    rng = np.random.default_rng(0)

    picks = {sample_next_id(logits, temperature=0.01, rng=rng) for _ in range(100)}

    assert picks == {4}


def test_high_temperature_spreads_the_picks():
    logits = np.array([0.0, 0.0, 2.0, 1.0, 3.0])
    rng = np.random.default_rng(0)

    picks = [sample_next_id(logits, temperature=100.0, rng=rng) for _ in range(3000)]

    # Close to equal: each of the three allowed IDs gets roughly a third.
    for allowed_id in (END_ID, 3, 4):
        assert 0.25 < picks.count(allowed_id) / len(picks) < 0.42
