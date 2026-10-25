import numpy as np
import pytest
import torch

from src.rnn_model import RNNClassifier


def test_rnn_returns_binary_logits_and_probabilities():
    torch.manual_seed(0)
    model = RNNClassifier(input_size=4, hidden_size=8)
    sequences = torch.randn(3, 5, 4)

    logits = model(sequences)
    probabilities = model.predict_proba(sequences)

    assert logits.shape == (3,)
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_rnn_accepts_numpy_sequences():
    model = RNNClassifier(input_size=2)
    probabilities = model.predict_proba(np.ones((2, 3, 2), dtype=np.float32))

    assert probabilities.shape == (2, 2)


def test_rnn_rejects_wrong_input_shape():
    model = RNNClassifier(input_size=2)

    with pytest.raises(ValueError, match="shape"):
        model(torch.ones(2, 2))


def test_rnn_rejects_wrong_feature_count():
    model = RNNClassifier(input_size=2)

    with pytest.raises(ValueError, match="Expected 2 features"):
        model(torch.ones(2, 3, 1))


def test_rnn_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="dropout must be 0"):
        RNNClassifier(input_size=2, num_layers=1, dropout=0.2)
