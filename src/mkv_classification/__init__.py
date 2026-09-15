"""Research implementation for plug-in classification of interacting particles."""

from .classifier import classification_accuracy, classification_scores, predict
from .estimator import DriftModel, fit_drift, mean_estimated_drift
from .simulation import observe_paths, simulate_ips
from .splines import BSplineBasis

__all__ = [
    "BSplineBasis",
    "DriftModel",
    "classification_accuracy",
    "classification_scores",
    "fit_drift",
    "mean_estimated_drift",
    "observe_paths",
    "predict",
    "simulate_ips",
]

__version__ = "1.0.0"
