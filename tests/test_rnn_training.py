import numpy as np
import pytest
import torch

from src.rnn_model import RNNClassifier
from src.rnn_training import RNNTrainer, TrainingHistory


def _data():
    X = np.array(
        [
            [[0.0, 0.0], [0.1, 0.0]],
            [[0.0, 1.0], [0.1, 1.0]],
            [[1.0, 0.0], [0.9, 0.0]],
            [[1.0, 1.0], [0.9, 1.0]],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    return X, y


def test_fit_records_training_and_validation_loss():
    torch.manual_seed(0)
    X, y = _data()
    trainer = RNNTrainer(
        RNNClassifier(input_size=2, hidden_size=4),
        batch_size=2,
        epochs=3,
    )

    history = trainer.fit(X, y, X, y)

    assert isinstance(history, TrainingHistory)
    assert len(history.training_loss) == 3
    assert len(history.validation_loss) == 3
    assert all(np.isfinite(history.training_loss))
    assert all(np.isfinite(history.validation_loss))


def test_trained_model_returns_probabilities():
    X, y = _data()
    trainer = RNNTrainer(RNNClassifier(input_size=2), epochs=1)
    trainer.fit(X, y)

    probabilities = trainer.predict_proba(X)

    assert probabilities.shape == (4, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(4))


def test_validation_data_must_be_complete():
    X, y = _data()
    trainer = RNNTrainer(RNNClassifier(input_size=2), epochs=1)

    with pytest.raises(ValueError, match="provided together"):
        trainer.fit(X, y, X_validate=X)


def test_training_rejects_invalid_labels():
    X, y = _data()
    y[0] = 2
    trainer = RNNTrainer(RNNClassifier(input_size=2), epochs=1)

    with pytest.raises(ValueError, match="only 0 and 1"):
        trainer.fit(X, y)
