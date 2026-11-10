"""Training utilities for the standalone recurrent sports-betting model."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rnn_model import RNNClassifier


@dataclass(frozen=True)
class TrainingHistory:
    """Loss values recorded after each training epoch."""

    training_loss: list[float]
    validation_loss: list[float]


class RNNTrainer:
    """Fit an :class:`RNNClassifier` on sequence arrays."""

    def __init__(
        self,
        model: RNNClassifier,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 20,
        device: Optional[str] = None,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")

        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model.to(self.device)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validate: Optional[np.ndarray] = None,
        y_validate: Optional[np.ndarray] = None,
    ) -> TrainingHistory:
        """Train the model and optionally record validation loss."""

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        validation_data = None
        if (X_validate is None) != (y_validate is None):
            raise ValueError("X_validate and y_validate must be provided together.")
        if X_validate is not None and y_validate is not None:
            validation_data = self._make_tensors(X_validate, y_validate)

        training_losses: list[float] = []
        validation_losses: list[float] = []
        for _ in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            sample_count = 0
            for features, targets in train_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss_function(self.model(features), targets)
                loss.backward()
                self.optimizer.step()
                batch_size = features.shape[0]
                epoch_loss += loss.item() * batch_size
                sample_count += batch_size
            training_losses.append(epoch_loss / sample_count)

            if validation_data is not None:
                validation_losses.append(self._evaluate_loss(*validation_data))

        return TrainingHistory(
            training_loss=training_losses,
            validation_loss=validation_losses,
        )

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ``[away-win, home-win]`` probabilities."""

        self.model.eval()
        features = self._validate_features(X).to(self.device)
        return self.model.predict_proba(features)

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> DataLoader:
        features, targets = self._make_tensors(X, y)
        return DataLoader(
            TensorDataset(features, targets),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def _make_tensors(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._validate_features(X)
        targets = torch.as_tensor(y, dtype=torch.float32)
        if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("y must contain only 0 and 1.")
        return features, targets

    def _validate_features(self, X: np.ndarray) -> torch.Tensor:
        features = torch.as_tensor(X, dtype=torch.float32)
        if features.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, features).")
        if features.shape[-1] != self.model.input_size:
            raise ValueError(
                f"Expected {self.model.input_size} features, got {features.shape[-1]}."
            )
        if features.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        return features

    def _evaluate_loss(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> float:
        features = features.to(self.device)
        targets = targets.to(self.device)
        return float(self.loss_function(self.model(features), targets).item())
