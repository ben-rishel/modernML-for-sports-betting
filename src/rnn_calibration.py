"""Calibration evaluation for the standalone RNN experiment."""

from typing import Any, Dict, Sequence

import numpy as np
from calibra.errors import classwise_ece
from calibra.utils import bin_probabilities, get_classwise_bin_weights
from sklearn.metrics import accuracy_score


def calculate_calibration_score(
    y_true: Sequence[int] | np.ndarray,
    predicted_probabilities: np.ndarray,
    num_bins: int = 20,
) -> float:
    """Apply the project's populated-bin guard before calculating ECE."""

    y_true_array = np.asarray(y_true, dtype=int)
    probabilities = _validate_probabilities(predicted_probabilities)
    if y_true_array.shape[0] != probabilities.shape[0]:
        raise ValueError("y_true and predicted_probabilities must have the same length.")
    if num_bins < 1:
        raise ValueError("num_bins must be at least 1.")

    binned = bin_probabilities(probabilities, y_true_array, num_bins=num_bins)
    weights = get_classwise_bin_weights(binned)[0]
    populated_bins = sum(float(weight) > 0 for weight in weights)
    if populated_bins < 0.8 * num_bins:
        return 1.0
    return _classwise_ece_scalar(probabilities, y_true_array, num_bins)


def evaluate_rnn_calibration(
    trainer: Any,
    X_evaluate: np.ndarray,
    y_evaluate: Sequence[int] | np.ndarray,
    num_bins: int = 20,
) -> Dict[str, Any]:
    """Evaluate a trained RNN using the repository's calibration protocol."""

    probabilities = np.asarray(trainer.predict_proba(X_evaluate), dtype=float)
    probabilities = _validate_probabilities(probabilities)
    y_true = np.asarray(y_evaluate, dtype=int)
    if y_true.shape[0] != probabilities.shape[0]:
        raise ValueError("y_evaluate and predictions must have the same length.")

    binned = bin_probabilities(probabilities, y_true, num_bins=num_bins)
    weights = get_classwise_bin_weights(binned)[0]
    populated_bins = sum(float(weight) > 0 for weight in weights)
    predicted_labels = (probabilities[:, 1] >= 0.5).astype(int)
    return {
        "calibration_score": calculate_calibration_score(
            y_true, probabilities, num_bins=num_bins
        ),
        "raw_calibration_score": _classwise_ece_scalar(
            probabilities, y_true, num_bins
        ),
        "populated_bins": populated_bins,
        "calibration_guard_failed": populated_bins < 0.8 * num_bins,
        "accuracy": float(accuracy_score(y_true, predicted_labels)),
        "probabilities": probabilities.tolist(),
    }


def _validate_probabilities(predicted_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(predicted_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Predictions must have shape (n_samples, 2).")
    if probabilities.shape[0] == 0:
        raise ValueError("Predictions must contain at least one sample.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions must contain only finite values.")
    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Each prediction must contain probabilities summing to 1.")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predictions must be between 0 and 1.")
    return probabilities


def _classwise_ece_scalar(
    probabilities: np.ndarray, y_true: np.ndarray, num_bins: int
) -> float:
    result = classwise_ece(probabilities, y_true, num_bins)
    if isinstance(result, tuple):
        result = result[0]
    return float(result)
