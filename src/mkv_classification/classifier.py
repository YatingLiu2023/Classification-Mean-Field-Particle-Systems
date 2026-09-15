"""Numerically stable plug-in classification scores."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .estimator import DriftModel, mean_estimated_drift


def classification_scores(
    paths: np.ndarray,
    models: Sequence[DriftModel],
    *,
    total_time: float = 1.0,
    priors: Sequence[float] | None = None,
    clip: bool = True,
    path_chunk_size: int = 128,
    measure_chunk_size: int = 256,
) -> np.ndarray:
    """Compute log-scores directly; no exponential or softmax is needed."""
    paths = np.asarray(paths, dtype=float)
    if not models:
        raise ValueError("At least one fitted model is required.")
    if priors is None:
        priors_array = np.full(len(models), 1.0 / len(models))
    else:
        priors_array = np.asarray(priors, dtype=float)
    if priors_array.shape != (len(models),) or np.any(priors_array <= 0):
        raise ValueError("priors must be positive and match the number of models.")
    priors_array = priors_array / priors_array.sum()
    n_times = paths.shape[1] - 1
    dt = total_time / n_times
    increments = np.diff(paths, axis=1)
    columns = []
    for class_index, model in enumerate(models):
        estimated, _ = mean_estimated_drift(
            paths,
            model,
            clip=clip,
            path_chunk_size=path_chunk_size,
            measure_chunk_size=measure_chunk_size,
        )
        columns.append(
            np.sum(estimated * increments, axis=1)
            - 0.5 * dt * np.sum(estimated**2, axis=1)
            + np.log(priors_array[class_index])
        )
    return np.column_stack(columns)


def predict(paths: np.ndarray, models: Sequence[DriftModel], **score_kwargs: object) -> np.ndarray:
    """Predict zero-based class indices."""
    return np.argmax(classification_scores(paths, models, **score_kwargs), axis=1)


def classification_accuracy(
    paths_by_class: Sequence[np.ndarray],
    models: Sequence[DriftModel],
    **score_kwargs: object,
) -> tuple[float, np.ndarray]:
    """Return overall accuracy and a confusion matrix."""
    if len(paths_by_class) != len(models):
        raise ValueError("One evaluation path system is required per class.")
    all_paths = np.vstack(paths_by_class)
    truth = np.concatenate(
        [np.full(len(paths), class_index, dtype=int) for class_index, paths in enumerate(paths_by_class)]
    )
    predictions = predict(all_paths, models, **score_kwargs)
    confusion = np.zeros((len(models), len(models)), dtype=int)
    np.add.at(confusion, (truth, predictions), 1)
    return float(np.mean(predictions == truth)), confusion

