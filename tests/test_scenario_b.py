import numpy as np

from mkv_classification.randomness import make_rng
from mkv_classification.scenario_b import mean_drift, simulate_ips


def test_scenario_b_theta_zero_is_brownian_euler_path() -> None:
    first = simulate_ips(0.0, 12, make_rng(12), n_steps=10)
    second = simulate_ips(0.0, 12, make_rng(12), n_steps=10)
    assert np.array_equal(first, second)
    assert not np.allclose(first[:, -1], 0.0)


def test_scenario_b_mean_drift_matches_pair_average() -> None:
    x = np.array([-0.5, 0.2])
    measure = np.array([-1.0, 0.0, 1.0])
    theta = 3.0
    expected = theta * (
        0.25 + 0.75 * np.cos(x) ** 2 + np.mean(np.cos(measure) ** 2)
    )
    assert np.allclose(mean_drift(theta, x, measure), expected)

