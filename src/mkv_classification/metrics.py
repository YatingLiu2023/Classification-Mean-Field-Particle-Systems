"""Evaluation metrics on independent particle systems."""

from __future__ import annotations

import numpy as np

from .estimator import DriftModel, mean_estimated_drift
from .simulation import true_mean_drift_paths


def drift_mean_squared_error(
    paths: np.ndarray,
    measure_paths: np.ndarray,
    model: DriftModel,
    *,
    label: int,
    theta0: float,
    clip: bool = True,
    pair_chunk_size: int = 128,
) -> float:
    """Squared empirical drift error on data not used for fitting."""
    estimated, _ = mean_estimated_drift(
        paths, model, measure_paths=measure_paths, clip=clip
    )
    target = true_mean_drift_paths(
        paths,
        measure_paths,
        label,
        theta0,
        pair_chunk_size=pair_chunk_size,
    )
    return float(np.mean((estimated - target) ** 2))
