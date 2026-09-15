import numpy as np
import pytest

from mkv_classification.randomness import make_rng
from mkv_classification.simulation import observe_paths, simulate_ips


def test_simulation_is_reproducible_and_stream_specific() -> None:
    first = simulate_ips(1, 12, 1.0, make_rng(42, 1), n_steps=10)
    repeated = simulate_ips(1, 12, 1.0, make_rng(42, 1), n_steps=10)
    other = simulate_ips(1, 12, 1.0, make_rng(42, 2), n_steps=10)
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, other)


def test_observation_grid_must_divide_simulation_grid() -> None:
    paths = simulate_ips(2, 10, 1.0, make_rng(7), n_steps=12)
    assert observe_paths(paths, 4).shape == (10, 5)
    with pytest.raises(ValueError):
        observe_paths(paths, 5)

