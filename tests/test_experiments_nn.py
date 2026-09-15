import numpy as np

from mkv_classification.experiments_nn import _test_diagnostics


def test_neural_diagnostics_detect_missing_predicted_class() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 0, 0, 1, 1, 1])

    diagnostics = _test_diagnostics(labels, predictions, n_classes=3)

    assert diagnostics["predicted_class_count"] == 2
    assert diagnostics["missing_predicted_classes"] == "2"
    assert diagnostics["class_collapse"] == 1
    assert diagnostics["zero_recall_class_count"] == 1
    assert diagnostics["class_0_recall"] == 1.0
    assert diagnostics["class_1_recall"] == 0.5
    assert diagnostics["class_2_recall"] == 0.0
    assert diagnostics["confusion_matrix"] == "[[2,0,0],[1,1,0],[0,2,0]]"


def test_neural_diagnostics_do_not_relabel_low_accuracy_as_collapse() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([1, 1, 0, 0])

    diagnostics = _test_diagnostics(labels, predictions, n_classes=2)

    assert diagnostics["predicted_class_count"] == 2
    assert diagnostics["class_collapse"] == 0
    assert diagnostics["chance_or_worse"] == 1
