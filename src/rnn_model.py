"""PyTorch recurrent model used by the standalone RNN experiment."""

from typing import Optional

import numpy as np
import torch
from torch import nn


class RNNClassifier(nn.Module):
    """Binary classifier for chronological game-history sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be at least 1.")
        if hidden_size < 1:
            raise ValueError("hidden_size must be at least 1.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return one logit per sequence."""

        if sequences.ndim != 3:
            raise ValueError(
                "sequences must have shape (batch, timesteps, features)."
            )
        if sequences.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, got {sequences.shape[-1]}."
            )

        outputs, _ = self.rnn(sequences.float())
        return self.classifier(outputs[:, -1, :]).squeeze(-1)

    @torch.no_grad()
    def predict_proba(
        self,
        sequences: torch.Tensor | np.ndarray,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Return home-win probabilities with shape ``(samples, 2)``."""

        self.eval()
        if isinstance(sequences, np.ndarray):
            sequences = torch.from_numpy(sequences)
        if device is not None:
            self.to(device)
            sequences = sequences.to(device)
        probabilities = torch.sigmoid(self(sequences)).cpu().numpy()
        return np.column_stack((1.0 - probabilities, probabilities))
