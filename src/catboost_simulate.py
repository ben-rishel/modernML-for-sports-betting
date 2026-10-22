import json
import os
from typing import Dict, List, Tuple, Union

import numpy as np
from calibra.errors import classwise_ece
from sklearn.metrics import accuracy_score

from simulate import BettingSimulation


class CatBoostBettingSimulation(BettingSimulation):
    """Run the existing betting rules with a CatBoost classifier.

    Prediction averaging, value-bet detection, fixed stakes, and fractional
    Kelly staking are inherited from ``BettingSimulation``. CatBoost metrics
    and betting results are written to a dedicated output directory.
    """

    def __init__(
        self,
        optimal_features: List[str],
        optimal_hyperparameters: Dict[str, Union[str, int, float]],
        X_train_final,
        y_train_final: List[int],
        X_betting_simulation,
        y_betting_simulation: List[int],
        odds,
        output_dir: str = os.path.join(".", "data", "output", "catboost"),
        num_runs: int = 1,
    ):
        super().__init__(
            optimal_model="catboost",
            optimal_features=optimal_features,
            optimal_hyperparameters={"catboost": dict(optimal_hyperparameters)},
            X_train_final=X_train_final,
            y_train_final=y_train_final,
            X_betting_simulation=X_betting_simulation,
            y_betting_simulation=y_betting_simulation,
            odds=odds,
            metric="calibration",
            num_runs=num_runs,
        )
        if len(self.y_betting_simulation) != len(self.X_betting_simulation):
            raise ValueError("Betting features and outcomes must contain the same number of games.")
        if len(self.odds) != len(self.X_betting_simulation):
            raise ValueError("Betting features and odds must contain the same number of games.")
        if num_runs < 1:
            raise ValueError("num_runs must be at least 1.")

        self.output_dir = output_dir

    def train_model(self, random_state: int = 0):
        """Fit CatBoost on the final training data using the tuned parameters."""
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CatBoost is required for the CatBoost betting simulation."
            ) from exc

        model_kwargs = dict(self.optimal_hyperparameters["catboost"])
        model_kwargs.setdefault("loss_function", "Logloss")
        model_kwargs.setdefault("verbose", False)
        model_kwargs.setdefault("allow_writing_files", False)
        model_kwargs["random_seed"] = random_state

        model = CatBoostClassifier(**model_kwargs)
        model.fit(self.X_train_final, self.y_train_final)
        return model

    def record_metrics(self) -> Dict[str, float]:
        """Record CatBoost accuracy and calibration without touching parent outputs."""
        predicted_labels = (self.prediction_array[:, 1] >= 0.5).astype(int)
        metrics = {
            "calibration_score": float(
                classwise_ece(self.prediction_array, self.y_betting_simulation)
            ),
            "accuracy": float(
                accuracy_score(self.y_betting_simulation, predicted_labels)
            ),
        }

        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(
            self.output_dir, "catboost_betting_simulation_metrics.json"
        )
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)
        return metrics

    def save_results(
        self,
        rule: str,
        betting_history: Dict[int, Dict[str, Union[float, bool]]],
        final_balance: float,
        test: bool = False,
    ) -> Tuple[Dict[str, Union[float, int]], Dict[str, Union[float, int]]]:
        """Save simulation summaries in the CatBoost-only output directory."""
        roi = 100 * (final_balance - self.INITIAL_BANKROLL) / self.INITIAL_BANKROLL
        num_games = self.prediction_array.shape[0]
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
            "bet_percentage": round(100 * bets_made / num_games, 2) if num_games else 0.0,
            "win_percentage": round(100 * bets_won / bets_made, 2) if bets_made else 0.0,
        }

        if not test:
            os.makedirs(self.output_dir, exist_ok=True)
            for suffix, results in (
                ("betting_results", betting_results),
                ("betting_summary", betting_summary),
            ):
                output_path = os.path.join(self.output_dir, f"{rule}_{suffix}.json")
                with open(output_path, "w", encoding="utf-8") as file:
                    json.dump(results, file, indent=4)

        return betting_results, betting_summary