"""Betting simulation adapter for the standalone RNN experiment."""

import json
import os
from typing import Dict, List, Tuple, TypedDict, Union

import numpy as np
from calibra.errors import classwise_ece
from sklearn.metrics import accuracy_score

from rnn_model import RNNClassifier
from rnn_training import RNNTrainer
from simulate import BettingSimulation


class RNNModelParameters(TypedDict):
    hidden_size: int
    num_layers: int
    dropout: float


class RNNTrainingParameters(TypedDict):
    learning_rate: float
    batch_size: int
    epochs: int


class BettingSide(TypedDict):
    bet: bool
    win: bool


BettingHistory = Dict[int, Dict[str, BettingSide]]


class RNNBettingSimulation(BettingSimulation):
    """Use RNN sequence arrays with the repository's existing betting rules."""

    def __init__(
        self,
        model_input_size: int,
        model_hyperparameters: RNNModelParameters,
        training_hyperparameters: RNNTrainingParameters,
        X_train_final: np.ndarray,
        y_train_final: List[int],
        X_betting_simulation: np.ndarray,
        y_betting_simulation: List[int],
        odds,
        output_dir: str = os.path.join(".", "data", "output", "rnn"),
        num_runs: int = 1,
    ):
        if len(X_train_final) != len(y_train_final):
            raise ValueError("Training sequences and outcomes must have the same length.")
        if len(X_betting_simulation) != len(y_betting_simulation):
            raise ValueError("Betting sequences and outcomes must have the same length.")
        if len(X_betting_simulation) != len(odds):
            raise ValueError("Betting sequences and odds must have the same length.")
        if num_runs < 1:
            raise ValueError("num_runs must be at least 1.")

        self.model_input_size = model_input_size
        self.model_hyperparameters = model_hyperparameters
        self.training_hyperparameters = training_hyperparameters
        self.X_train_final = np.asarray(X_train_final, dtype=np.float32)
        self.y_train_final = list(y_train_final)
        self.X_betting_simulation = np.asarray(
            X_betting_simulation, dtype=np.float32
        )
        self.y_betting_simulation = list(y_betting_simulation)
        self.odds = odds
        self.num_runs = num_runs
        self.output_dir = output_dir
        self.metric = "calibration"
        self.model_name = "rnn"

    def train_model(self, random_state: int = 0) -> RNNTrainer:
        """Create and fit one seeded RNN trainer."""

        np.random.seed(random_state)
        import torch

        torch.manual_seed(random_state)
        model = RNNClassifier(
            input_size=self.model_input_size,
            **self.model_hyperparameters,
        )
        trainer = RNNTrainer(model, **self.training_hyperparameters)
        trainer.fit(self.X_train_final, np.asarray(self.y_train_final))
        return trainer

    def generate_predictions(self) -> None:
        """Average predicted probabilities across independently seeded runs."""

        probability_sum = np.zeros((len(self.X_betting_simulation), 2))
        for random_state in range(self.num_runs):
            trainer = self.train_model(random_state)
            probability_sum += trainer.predict_proba(self.X_betting_simulation)
        self.prediction_array = probability_sum / self.num_runs

    def record_metrics(self) -> Dict[str, float]:
        """Record RNN accuracy and calibration in the RNN output directory."""

        predicted_labels = (self.prediction_array[:, 1] >= 0.5).astype(int)
        metrics = {
            "calibration_score": _classwise_ece_scalar(
                self.prediction_array, self.y_betting_simulation
            ),
            "accuracy": float(
                accuracy_score(self.y_betting_simulation, predicted_labels)
            ),
        }
        os.makedirs(self.output_dir, exist_ok=True)
        with open(
            os.path.join(self.output_dir, "rnn_betting_simulation_metrics.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics, file, indent=4)
        return metrics

    def save_results(
        self,
        rule: str,
        betting_history: BettingHistory,
        final_balance: float,
        test: bool = False,
    ) -> Tuple[Dict[str, Union[float, int]], Dict[str, Union[float, int]]]:
        """Save RNN betting summaries without changing parent-pipeline outputs."""

        roi = 100 * (final_balance - self.INITIAL_BANKROLL) / self.INITIAL_BANKROLL
        bets_made = 0
        bets_won = 0
        for game in betting_history.values():
            for side in ("home", "away"):
                if game[side]["bet"]:
                    bets_made += 1
                    if game[side]["win"]:
                        bets_won += 1

        betting_results = {
            "final_balance": float(final_balance),
            "ROI": round(roi, 2),
        }
        betting_summary = {
            "bets_made": bets_made,
            "bets_won": bets_won,
            "bet_percentage": round(
                100 * bets_made / len(self.prediction_array), 2
            )
            if len(self.prediction_array)
            else 0.0,
            "win_percentage": round(100 * bets_won / bets_made, 2)
            if bets_made
            else 0.0,
        }
        if not test:
            os.makedirs(self.output_dir, exist_ok=True)
            for suffix, results in (
                ("betting_results", betting_results),
                ("betting_summary", betting_summary),
            ):
                with open(
                    os.path.join(self.output_dir, f"{rule}_{suffix}.json"),
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(results, file, indent=4)
        return betting_results, betting_summary


def _classwise_ece_scalar(
    probabilities: np.ndarray, y_true: List[int]
) -> float:
    result = classwise_ece(probabilities, np.asarray(y_true, dtype=int))
    if isinstance(result, tuple):
        result = result[0]
    return float(result)
