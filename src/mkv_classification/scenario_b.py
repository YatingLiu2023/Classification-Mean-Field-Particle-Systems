"""Five-class Scenario B from the paper."""

from __future__ import annotations

import numpy as np


def pair_drift(theta: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return theta * (0.25 + 0.75 * np.cos(x) ** 2 + np.cos(y) ** 2)


def mean_drift(theta: float, x: np.ndarray, measure: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    measure = np.asarray(measure, dtype=float)
    return theta * (0.25 + 0.75 * np.cos(x) ** 2 + np.mean(np.cos(measure) ** 2))


def simulate_ips(
    theta: float,
    n_particles: int,
    rng: np.random.Generator,
    *,
    total_time: float = 1.0,
    n_steps: int = 100,
    initial_value: float = 0.0,
) -> np.ndarray:
    if n_particles < 2 or n_steps < 1 or total_time <= 0:
        raise ValueError("n_particles >= 2, n_steps >= 1, and total_time > 0 are required.")
    dt = total_time / n_steps
    paths = np.empty((n_particles, n_steps + 1), dtype=float)
    paths[:, 0] = initial_value
    for m in range(n_steps):
        state = paths[:, m]
        paths[:, m + 1] = (
            state + dt * mean_drift(theta, state, state)
            + np.sqrt(dt) * rng.standard_normal(n_particles)
        )
    return paths


def true_mean_drift_paths(
    paths: np.ndarray,
    measure_paths: np.ndarray,
    theta: float,
) -> np.ndarray:
    paths = np.asarray(paths, dtype=float)
    measure_paths = np.asarray(measure_paths, dtype=float)
    if paths.ndim != 2 or measure_paths.ndim != 2 or paths.shape[1] != measure_paths.shape[1]:
        raise ValueError("path and measure systems must share the same grid.")
    output = np.empty((paths.shape[0], paths.shape[1] - 1), dtype=float)
    for m in range(paths.shape[1] - 1):
        output[:, m] = mean_drift(theta, paths[:, m], measure_paths[:, m])
    return output

