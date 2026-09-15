import numpy as np
import mkv_classification.neural_network as neural_network

from mkv_classification.neural_network import (
    NetworkParameters,
    fit_network,
    stack_labeled_paths,
    tune_and_evaluate,
)
from mkv_classification.randomness import make_rng


def test_stack_labeled_paths_preserves_complete_systems() -> None:
    systems = [np.full((3, 2), -1.0), np.full((4, 2), 1.0)]
    features, labels = stack_labeled_paths(systems)
    assert features.shape == (7, 2)
    assert np.array_equal(labels, np.array([0, 0, 0, 1, 1, 1, 1]))


def test_multiclass_network_training_is_finite() -> None:
    rng = make_rng(4)
    features = np.vstack(
        [rng.normal(-2, 0.2, (20, 2)), rng.normal(0, 0.2, (20, 2)), rng.normal(2, 0.2, (20, 2))]
    )
    labels = np.repeat(np.arange(3), 20)
    fitted = fit_network(
        features[:45], labels[:45], features[45:], labels[45:],
        n_classes=3, parameters=NetworkParameters(1, 16, 0.01), rng=make_rng(6),
        epochs=20, batch_size=16, patience=5, sparsity_budget=500,
    )
    predictions = fitted.predict(features)
    assert predictions.shape == (60,)
    assert 0.0 <= fitted.validation_accuracy <= 1.0
    assert np.isfinite(fitted.validation_loss)
    assert np.isfinite(fitted.feature_mean).all()


def test_tuning_keeps_independent_system_roles(monkeypatch) -> None:
    calls = []

    class StubFit:
        validation_accuracy = 0.75
        validation_loss = 0.5
        best_epoch = 1

        def predict(self, features):
            return np.zeros(len(features), dtype=int)

    def fake_fit(train_features, train_labels, validation_features, validation_labels, **kwargs):
        calls.append((train_features.copy(), validation_features.copy()))
        return StubFit()

    monkeypatch.setattr(neural_network, "fit_network", fake_fit)
    train = [np.full((3, 2), 1.0), np.full((3, 2), 2.0)]
    validation = [np.full((4, 2), 11.0), np.full((4, 2), 12.0)]
    test = [np.full((5, 2), 21.0), np.full((5, 2), 22.0)]

    result = tune_and_evaluate(
        train, validation, test, rng_factory=lambda stream: make_rng(7, stream),
        trials=2, tune_epochs=1, max_epochs=1, batch_size=4, patience=1,
        sparsity_budget=500,
    )

    assert len(calls) == 3  # two tuning trials and one fresh full training
    assert all(np.max(train_features) == 2.0 for train_features, _ in calls)
    assert all(np.min(validation_features) == 11.0 for _, validation_features in calls)
    assert result.accuracy == 0.5
    assert result.validation_loss == 0.5
    assert result.selected_trial == 0


def test_tuning_keeps_first_accuracy_winner_and_does_not_rescue(monkeypatch) -> None:
    outcomes = [
        (0.80, 0.90),  # Short-run accuracy winner.
        (0.80, 0.20),  # Tied accuracy with better loss: intentionally ignored.
        (0.50, 0.70),  # The sole fresh full training may fail.
    ]

    class StubFit:
        def __init__(self, accuracy, loss):
            self.validation_accuracy = accuracy
            self.validation_loss = loss
            self.best_epoch = 1

        def predict(self, features):
            return np.zeros(len(features), dtype=int)

    def fake_fit(*args, **kwargs):
        return StubFit(*outcomes.pop(0))

    monkeypatch.setattr(neural_network, "fit_network", fake_fit)
    systems = [np.zeros((3, 2)), np.ones((3, 2))]
    result = tune_and_evaluate(
        systems, systems, systems,
        rng_factory=lambda stream: make_rng(8, stream),
        trials=2, tune_epochs=1, max_epochs=1, batch_size=4, patience=1,
        sparsity_budget=500,
    )

    assert result.validation_accuracy == 0.50
    assert result.selected_trial == 0
    assert outcomes == []
