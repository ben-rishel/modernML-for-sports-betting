import numpy as np
import pytest

from src.rnn_calibration import (
    calculate_calibration_score,
    evaluate_rnn_calibration,
)


def test_calibration_score_is_finite():
    score = calculate_calibration_score(
        [0, 0, 1, 1],
        np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]]),
    )

    assert isinstance(score, float)
    assert score >= 0


def test_evaluation_reports_accuracy_and_probabilities():
    class FakeTrainer:
        def predict_proba(self, X):
            return np.array([[0.8, 0.2], [0.1, 0.9]])

    result = evaluate_rnn_calibration(
        FakeTrainer(), np.zeros((2, 1, 1)), [0, 1], num_bins=2
    )

    assert result["accuracy"] == 1.0
    assert result["probabilities"] == [[0.8, 0.2], [0.1, 0.9]]
    assert result["populated_bins"] == 2


def test_calibration_rejects_invalid_probability_shape():
    with pytest.raises(ValueError, match="shape"):
        calculate_calibration_score([0], np.array([[0.5]]))


def test_calibration_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="same length"):
        calculate_calibration_score([0], np.array([[0.5, 0.5], [0.4, 0.6]]))
