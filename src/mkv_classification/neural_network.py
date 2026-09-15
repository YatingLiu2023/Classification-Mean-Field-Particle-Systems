"""Deterministic NumPy direct-neural-network benchmark used in the paper."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NetworkParameters:
    hidden_layers: int
    hidden_width: int
    learning_rate: float


class DenseClassifier:
    def __init__(
        self,
        input_dimension: int,
        n_classes: int,
        parameters: NetworkParameters,
        rng: np.random.Generator,
    ) -> None:
        dimensions = [input_dimension] + [parameters.hidden_width] * parameters.hidden_layers + [n_classes]
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / left), size=(left, right))
            for left, right in zip(dimensions[:-1], dimensions[1:])
        ]
        self.biases = [np.zeros((1, right), dtype=float) for right in dimensions[1:]]

    def forward(self, features: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        activations = [features]
        preactivations: list[np.ndarray] = []
        for weights, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = activations[-1] @ weights + bias
            preactivations.append(z)
            activations.append(np.maximum(z, 0.0))
        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        preactivations.append(logits)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        activations.append(probabilities)
        return activations, preactivations

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(features)[0][-1], axis=1)


@dataclass
class FittedNetwork:
    network: DenseClassifier
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    validation_accuracy: float
    validation_loss: float
    best_epoch: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.network.predict((features - self.feature_mean) / self.feature_scale)


@dataclass(frozen=True)
class NeuralEvaluation:
    """One untouched final-test evaluation of a single selected network."""

    accuracy: float
    parameters: NetworkParameters
    best_epoch: int
    validation_accuracy: float
    validation_loss: float
    selected_trial: int
    test_labels: np.ndarray
    test_predictions: np.ndarray


def _project_parameters(arrays: list[np.ndarray], sparsity_budget: int) -> None:
    for array in arrays:
        np.clip(array, -1.0, 1.0, out=array)
    if sparsity_budget <= 0:
        return
    flattened = np.concatenate([np.abs(array).reshape(-1) for array in arrays])
    if flattened.size <= sparsity_budget:
        return
    threshold = np.partition(flattened, -sparsity_budget)[-sparsity_budget]
    for array in arrays:
        array[np.abs(array) < threshold] = 0.0


def fit_network(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    n_classes: int,
    parameters: NetworkParameters,
    rng: np.random.Generator,
    epochs: int,
    batch_size: int,
    patience: int | None,
    sparsity_budget: int,
) -> FittedNetwork:
    feature_mean = train_features.mean(axis=0)
    feature_scale = train_features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    X = (train_features - feature_mean) / feature_scale
    X_validation = (validation_features - feature_mean) / feature_scale
    network = DenseClassifier(X.shape[1], n_classes, parameters, rng)
    arrays = network.weights + network.biases
    first_moments = [np.zeros_like(array) for array in arrays]
    second_moments = [np.zeros_like(array) for array in arrays]
    best: tuple[float, float, int, list[np.ndarray]] | None = None
    wait = 0
    iteration = 0

    for epoch in range(epochs):
        permutation = rng.permutation(len(X))
        for indices in np.array_split(
            permutation, max(1, math.ceil(len(X) / batch_size))
        ):
            batch = X[indices]
            labels = train_labels[indices]
            activations, preactivations = network.forward(batch)
            delta = activations[-1].copy()
            delta[np.arange(len(indices)), labels] -= 1.0
            delta /= len(indices)
            weight_gradients: list[np.ndarray] = [None] * len(network.weights)  # type: ignore[list-item]
            bias_gradients: list[np.ndarray] = [None] * len(network.biases)  # type: ignore[list-item]
            for layer in range(len(network.weights) - 1, -1, -1):
                weight_gradients[layer] = activations[layer].T @ delta
                bias_gradients[layer] = delta.sum(axis=0, keepdims=True)
                if layer:
                    delta = (delta @ network.weights[layer].T) * (preactivations[layer - 1] > 0.0)
            iteration += 1
            for index, (array, gradient) in enumerate(
                zip(arrays, weight_gradients + bias_gradients)
            ):
                first_moments[index] = 0.9 * first_moments[index] + 0.1 * gradient
                second_moments[index] = 0.999 * second_moments[index] + 0.001 * gradient**2
                corrected_first = first_moments[index] / (1.0 - 0.9**iteration)
                corrected_second = second_moments[index] / (1.0 - 0.999**iteration)
                array -= parameters.learning_rate * corrected_first / (
                    np.sqrt(corrected_second) + 1e-8
                )
            _project_parameters(arrays, sparsity_budget)

        validation_probabilities = network.forward(X_validation)[0][-1]
        validation_accuracy = float(
            np.mean(np.argmax(validation_probabilities, axis=1) == validation_labels)
        )
        selected_probabilities = validation_probabilities[
            np.arange(len(validation_labels)), validation_labels
        ]
        validation_loss = float(
            -np.log(np.maximum(selected_probabilities, np.finfo(float).tiny)).mean()
        )
        # The legacy training protocol checkpoints validation accuracy only.
        # In particular, cross-entropy is diagnostic and is not a rescue
        # criterion when the classifier has collapsed.
        improved = best is None or validation_accuracy > best[1] + 1e-12
        if improved:
            best = (
                validation_loss,
                validation_accuracy,
                epoch + 1,
                [array.copy() for array in arrays],
            )
            wait = 0
        else:
            wait += 1
        if patience is not None and wait >= patience:
            break
    assert best is not None
    for array, saved in zip(arrays, best[3]):
        array[:] = saved
    return FittedNetwork(
        network, feature_mean, feature_scale,
        validation_accuracy=best[1], validation_loss=best[0], best_epoch=best[2],
    )


def stack_labeled_paths(paths_by_class: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack complete class-specific systems without mixing their data roles."""
    features = np.vstack(paths_by_class)
    labels = np.concatenate(
        [np.full(len(paths), index) for index, paths in enumerate(paths_by_class)]
    )
    return features, labels


def tune_and_evaluate(
    train_paths_by_class: list[np.ndarray],
    validation_measures_by_class: list[np.ndarray],
    test_paths_by_class: list[np.ndarray],
    *,
    rng_factory,
    trials: int,
    tune_epochs: int,
    max_epochs: int,
    batch_size: int,
    patience: int,
    sparsity_budget: int,
) -> NeuralEvaluation:
    """Select one trial on empirical-measure validation and test it once.

    This deliberately preserves the original single-finalist behavior: the
    short-run validation-accuracy winner is reinitialized once for full
    training.  There is no runner-up, loss-based rescue, or failed-run rerun.
    """
    n_classes = len(train_paths_by_class)
    if len(validation_measures_by_class) != n_classes or len(test_paths_by_class) != n_classes:
        raise ValueError("Train, validation, and test must contain the same classes.")
    train_features, train_labels = stack_labeled_paths(train_paths_by_class)
    validation_features, validation_labels = stack_labeled_paths(validation_measures_by_class)
    candidates: list[tuple[FittedNetwork, NetworkParameters, int]] = []
    search_rng = rng_factory(1)
    for trial in range(trials):
        parameters = NetworkParameters(
            hidden_layers=int(search_rng.integers(1, 5)),
            hidden_width=int(search_rng.integers(1, 9)) * 16,
            learning_rate=float(10.0 ** search_rng.uniform(-4.0, -2.0)),
        )
        fitted = fit_network(
            train_features, train_labels, validation_features, validation_labels,
            n_classes=n_classes, parameters=parameters, rng=rng_factory(10 + trial),
            epochs=tune_epochs, batch_size=batch_size, patience=None,
            sparsity_budget=sparsity_budget,
        )
        candidates.append((fitted, parameters, trial))
    # Match the original selection rule: maximum short-run validation
    # accuracy, with the first trial retained when accuracies tie.
    selected_trial = max(
        range(len(candidates)), key=lambda index: candidates[index][0].validation_accuracy
    )
    parameters = candidates[selected_trial][1]
    # Match the original final-training behavior: start once from a fresh
    # initialization.  A poor initialization remains part of the Monte Carlo
    # result rather than being replaced by the successful tuning checkpoint.
    final = fit_network(
        train_features, train_labels, validation_features, validation_labels,
        n_classes=n_classes, parameters=parameters, rng=rng_factory(1000),
        epochs=max_epochs, batch_size=batch_size, patience=patience,
        sparsity_budget=sparsity_budget,
    )
    test_features, test_labels = stack_labeled_paths(test_paths_by_class)
    test_predictions = final.predict(test_features)
    return NeuralEvaluation(
        accuracy=float(np.mean(test_predictions == test_labels)),
        parameters=parameters,
        best_epoch=final.best_epoch,
        validation_accuracy=final.validation_accuracy,
        validation_loss=final.validation_loss,
        selected_trial=selected_trial,
        test_labels=test_labels,
        test_predictions=test_predictions,
    )
