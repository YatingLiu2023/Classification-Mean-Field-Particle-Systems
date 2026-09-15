"""Deterministic, order-independent random streams."""

from __future__ import annotations

import numpy as np


def make_rng(base_seed: int, *stream_ids: int) -> np.random.Generator:
    """Create a reproducible stream without depending on loop execution order."""
    entropy = [int(base_seed), *(int(value) for value in stream_ids)]
    seed_sequence = np.random.SeedSequence(entropy)
    return np.random.Generator(np.random.PCG64(seed_sequence))
