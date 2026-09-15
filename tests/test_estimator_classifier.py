import numpy as np

from mkv_classification.classifier import classification_accuracy, classification_scores
from mkv_classification.estimator import DriftModel, SolverDiagnostics, fit_drift, mean_estimated_drift
from mkv_classification.randomness import make_rng
from mkv_classification.simulation import simulate_ips
from mkv_classification.splines import BSplineBasis


def _diagnostics(dimension: int) -> SolverDiagnostics:
    return SolverDiagnostics(0.0, False, 0.0, 1.0, dimension, dimension, 1.0, 0.0)


def test_pairwise_clipping_precedes_measure_average() -> None:
    basis = BSplineBasis(D=1, degree=0, A=1.0)
    paths = np.array([[0.0, 0.0]])
    # Two measure particles have basis values 1 and 1. The coefficient is 10,
    # so paper-level clipping produces sqrt(log(4)), not an unclipped value 10.
    model = DriftModel(
        basis=basis,
        coefficients=np.array([[10.0]]),
        measure_basis=np.ones((1, 2, 1)),
        n_train=4,
        diagnostics=_diagnostics(1),
        train_outside_rate=0.0,
    )
    clipped, _ = mean_estimated_drift(paths, model, clip=True)
    unclipped, _ = mean_estimated_drift(paths, model, clip=False)
    assert np.allclose(clipped, np.sqrt(np.log(4.0)))
    assert np.allclose(unclipped, 10.0)


def test_auxiliary_measure_can_replace_training_measure_for_drift_error() -> None:
    basis = BSplineBasis(D=2, degree=0, A=1.0)
    paths = np.array([[-0.5, -0.5]])
    training_measure_basis = np.array([[[1.0, 0.0], [1.0, 0.0]]])
    model = DriftModel(
        basis=basis,
        coefficients=np.array([[1.0, 2.0], [3.0, 4.0]]),
        measure_basis=training_measure_basis,
        n_train=100,
        diagnostics=_diagnostics(4),
        train_outside_rate=0.0,
    )
    training_average, _ = mean_estimated_drift(paths, model, clip=False)
    auxiliary_average, _ = mean_estimated_drift(
        paths, model, measure_paths=np.array([[0.5, 0.5], [0.5, 0.5]]), clip=False
    )
    assert np.allclose(training_average, 1.0)
    assert np.allclose(auxiliary_average, 2.0)


def test_small_pipeline_has_finite_scores() -> None:
    train_paths = []
    train_measures = []
    test_paths = []
    for label in (1, 2):
        train_paths.append(simulate_ips(label, 20, 1.0, make_rng(9, label, 1), n_steps=10))
        train_measures.append(simulate_ips(label, 20, 1.0, make_rng(9, label, 2), n_steps=10))
        test_paths.append(simulate_ips(label, 12, 1.0, make_rng(9, label, 3), n_steps=10))
    models = [fit_drift(train_paths[i], train_measures[i], D=2) for i in range(2)]
    scores = classification_scores(np.vstack(test_paths), models)
    accuracy, confusion = classification_accuracy(test_paths, models)
    assert scores.shape == (24, 2)
    assert np.isfinite(scores).all()
    assert 0.0 <= accuracy <= 1.0
    assert confusion.sum() == 24
    assert max(model.diagnostics.kkt_residual for model in models) < 1e-7
