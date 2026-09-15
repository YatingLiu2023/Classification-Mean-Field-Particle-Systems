"""Scenario-A interacting-particle simulation and true drift functions."""

from __future__ import annotations

import numpy as np


def pair_drift(label: int, x: np.ndarray, y: np.ndarray, theta0: float) -> np.ndarray:
    """Evaluate the class-specific interaction function b*(x, y)."""
    if label == 1:
        return theta0 * (0.25 + 0.75 * np.cos(x) ** 2 + np.cos(y) ** 2)
    if label == 2:
        delta = x - y
        return 2.0 * delta * np.exp(-(delta**2)) - 8.0 * delta * np.exp(-2.0 * delta**2)
    raise ValueError(f"Scenario A has labels 1 and 2, received {label}.")


def mean_interaction_drift(
    x: np.ndarray,
    measure: np.ndarray,
    label: int,
    theta0: float,
    *,
    chunk_size: int = 128,
) -> np.ndarray:
    """Evaluate the empirical-measure drift with bounded working memory."""
    x = np.asarray(x, dtype=float)
    measure = np.asarray(measure, dtype=float)
    if x.ndim != 1 or measure.ndim != 1 or measure.size == 0:
        raise ValueError("x and measure must be non-empty one-dimensional arrays.")
    if label == 1:
        return theta0 * (0.25 + 0.75 * np.cos(x) ** 2 + np.mean(np.cos(measure) ** 2))
    if label != 2:
        raise ValueError(f"Scenario A has labels 1 and 2, received {label}.")
    output = np.empty_like(x)
    for start in range(0, x.size, chunk_size):
        stop = min(start + chunk_size, x.size)
        output[start:stop] = pair_drift(
            2, x[start:stop, None], measure[None, :], theta0
        ).mean(axis=1)
    return output


def simulate_ips(
    label: int,
    n_particles: int,
    theta0: float,
    rng: np.random.Generator,
    *,
    total_time: float = 1.0,
    n_steps: int = 100,
    initial_value: float = 0.0,
    pair_chunk_size: int = 128,
) -> np.ndarray:
    """Simulate one complete interacting particle system by Euler--Maruyama."""
    if n_particles < 2 or n_steps < 1 or total_time <= 0:
        raise ValueError("n_particles >= 2, n_steps >= 1, and total_time > 0 are required.")
    dt = total_time / n_steps
    paths = np.empty((n_particles, n_steps + 1), dtype=float)
    paths[:, 0] = initial_value
    noise_scale = np.sqrt(dt)
    for time_index in range(n_steps):
        state = paths[:, time_index]
        drift = mean_interaction_drift(
            state,
            state,
            label,
            theta0,
            chunk_size=pair_chunk_size,
        )
        paths[:, time_index + 1] = (
            state + dt * drift + noise_scale * rng.standard_normal(n_particles)
        )
    return paths


def observe_paths(paths: np.ndarray, n_observations: int) -> np.ndarray:
    """Subsample a finely simulated system on a regular observation grid."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2:
        raise ValueError("paths must have shape (n_particles, n_simulation_steps + 1).")
    n_simulation_steps = paths.shape[1] - 1
    if n_observations < 1 or n_simulation_steps % n_observations:
        raise ValueError("n_observations must divide the simulation step count exactly.")
    return paths[:, :: n_simulation_steps // n_observations].copy()


def true_mean_drift_paths(
    paths: np.ndarray,
    measure_paths: np.ndarray,
    label: int,
    theta0: float,
    *,
    pair_chunk_size: int = 128,
) -> np.ndarray:
    """Evaluate the true empirical-measure drift along observed paths."""
    paths = np.asarray(paths, dtype=float)
    measure_paths = np.asarray(measure_paths, dtype=float)
    if paths.ndim != 2 or measure_paths.ndim != 2 or paths.shape[1] != measure_paths.shape[1]:
        raise ValueError("path and measure arrays must share the same observation grid.")
    n_times = paths.shape[1] - 1
    output = np.empty((paths.shape[0], n_times), dtype=float)
    for m in range(n_times):
        output[:, m] = mean_interaction_drift(
            paths[:, m],
            measure_paths[:, m],
            label,
            theta0,
            chunk_size=pair_chunk_size,
        )
    return output

