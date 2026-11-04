"""CatBoost-style RNN experiment using the parent pipeline's datasets."""

import json
import os
from typing import Dict, List

import numpy as np
from calibra.errors import classwise_ece
from calibra.utils import bin_probabilities, get_classwise_bin_weights
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score

from feature_selection import FeatureSelector
from rnn_model import RNNClassifier
from rnn_simulate import RNNBettingSimulation
from rnn_training import RNNTrainer
from sequential_data import build_row_sequences


class RNNCalibrationHPO:
    """Tune RNN parameters using the same validation protocol as CatBoost."""

    def __init__(
        self,
        X_train,
        X_validate,
        y_train: List[int],
        y_validate: List[int],
        num_trials: int = 10,
        num_scoring_runs: int = 5,
        evaluations_per_trial: int = 64,
    ):
        self.X_train = X_train.copy()
        self.X_validate = X_validate.copy()
        self.y_train = list(y_train)
        self.y_validate = list(y_validate)
        self.num_trials = num_trials
        self.num_scoring_runs = num_scoring_runs
        self.evaluations_per_trial = evaluations_per_trial
        if min(num_trials, num_scoring_runs, evaluations_per_trial) < 1:
            raise ValueError(
                "HPO trials, scoring runs, and evaluations per trial must be positive."
            )

    @staticmethod
    def calculate_calibration_score(y_true, probabilities, num_bins: int = 20) -> float:
        y_true = np.asarray(y_true, dtype=int)
        probabilities = np.asarray(probabilities, dtype=float)
        binned = bin_probabilities(probabilities, y_true, num_bins=num_bins)
        weights = get_classwise_bin_weights(binned)[0]
        if sum(float(weight) > 0 for weight in weights) < 0.8 * num_bins:
            return 1.0
        return float(classwise_ece(probabilities, y_true, num_bins))

    def define_search_space(self) -> Dict[str, object]:
        return {
            "hidden_size": hp.choice("hidden_size", [16, 32, 64]),
            "num_layers": hp.choice("num_layers", [1, 2]),
            "learning_rate": hp.loguniform(
                "learning_rate", np.log(0.0001), np.log(0.01)
            ),
            "batch_size": hp.choice("batch_size", [16, 32, 64]),
            "epochs": hp.choice("epochs", [10, 20, 40]),
            "sequence_length": hp.choice("sequence_length", [5, 10, 15]),
        }

    @staticmethod
    def _decode(best: Dict[str, object]) -> Dict[str, object]:
        return {
            "hidden_size": [16, 32, 64][best["hidden_size"]],
            "num_layers": [1, 2][best["num_layers"]],
            "dropout": 0.0,
            "learning_rate": float(best["learning_rate"]),
            "batch_size": [16, 32, 64][best["batch_size"]],
            "epochs": [10, 20, 40][best["epochs"]],
            "sequence_length": [5, 10, 15][best["sequence_length"]],
        }

    def _fit_and_score(self, params: Dict[str, object], seed: int) -> float:
        sequence_length = int(params["sequence_length"])
        X_train, y_train, _ = build_row_sequences(
            self.X_train, self.y_train, sequence_length
        )
        X_validate, y_validate, _ = build_row_sequences(
            self.X_validate, self.y_validate, sequence_length
        )
        np.random.seed(seed)
        import torch

        torch.manual_seed(seed)
        trainer = RNNTrainer(
            RNNClassifier(
                input_size=X_train.shape[-1],
                hidden_size=int(params["hidden_size"]),
                num_layers=int(params["num_layers"]),
                dropout=float(params["dropout"]),
            ),
            learning_rate=float(params["learning_rate"]),
            batch_size=int(params["batch_size"]),
            epochs=int(params["epochs"]),
        )
        trainer.fit(X_train, y_train)
        return self.calculate_calibration_score(
            y_validate, trainer.predict_proba(X_validate)
        )

    def run_single_hpo_trial(self, trial_index: int) -> Dict[str, object]:
        trials = Trials()
        best = fmin(
            fn=lambda params: {
                "loss": self._fit_and_score(
                    {**params, "dropout": 0.0}, trial_index
                ),
                "status": STATUS_OK,
            },
            space=self.define_search_space(),
            algo=tpe.suggest,
            max_evals=self.evaluations_per_trial,
            trials=trials,
            rstate=np.random.default_rng(trial_index),
        )
        parameters = self._decode(best)
        scores = [
            self._fit_and_score(parameters, seed)
            for seed in range(self.num_scoring_runs)
        ]
        return {"hyperparameters": parameters, "score": float(np.mean(scores))}

    def run_hpo(self) -> Dict[str, object]:
        self.hpo_results = [
            self.run_single_hpo_trial(index) for index in range(self.num_trials)
        ]
        return min(self.hpo_results, key=lambda result: result["score"])[
            "hyperparameters"
        ]


def evaluate_rnn(
    X_train,
    X_validate,
    y_train: List[int],
    y_validate: List[int],
    num_runs: int = 1,
    **parameters,
) -> dict:
    """Average seeded RNN predictions using the CatBoost calibration protocol."""

    X_train, y_train, _ = build_row_sequences(
        X_train, y_train, int(parameters["sequence_length"])
    )
    X_validate, y_validate, _ = build_row_sequences(
        X_validate, y_validate, int(parameters["sequence_length"])
    )
    probability_sum = np.zeros((len(y_validate), 2), dtype=float)
    calibration_scores = []
    raw_scores = []
    populated_bins = []
    accuracies = []
    for seed in range(num_runs):
        np.random.seed(seed)
        import torch

        torch.manual_seed(seed)
        trainer = RNNTrainer(
            RNNClassifier(
                input_size=X_train.shape[-1],
                hidden_size=int(parameters["hidden_size"]),
                num_layers=int(parameters["num_layers"]),
                dropout=float(parameters["dropout"]),
            ),
            learning_rate=float(parameters["learning_rate"]),
            batch_size=int(parameters["batch_size"]),
            epochs=int(parameters["epochs"]),
        )
        trainer.fit(X_train, y_train)
        probabilities = trainer.predict_proba(X_validate)
        probability_sum += probabilities
        binned = bin_probabilities(probabilities, y_validate, num_bins=20)
        weights = get_classwise_bin_weights(binned)[0]
        populated = sum(float(weight) > 0 for weight in weights)
        populated_bins.append(populated)
        raw_scores.append(float(classwise_ece(probabilities, y_validate, 20)))
        calibration_scores.append(
            RNNCalibrationHPO.calculate_calibration_score(y_validate, probabilities)
        )
        accuracies.append(accuracy_score(y_validate, probabilities[:, 1] >= 0.5))
    return {
        "calibration_score": float(np.mean(calibration_scores)),
        "raw_calibration_score": float(np.mean(raw_scores)),
        "calibration_scores_per_run": calibration_scores,
        "raw_calibration_scores_per_run": raw_scores,
        "populated_bins_per_run": populated_bins,
        "calibration_guard_failures": sum(value < 16 for value in populated_bins),
        "accuracy": float(np.mean(accuracies)),
        "probabilities": (probability_sum / num_runs).tolist(),
    }


def run_rnn_calibration_side_experiment(
    experiment,
    output_dir: str = os.path.join(".", "data", "output", "rnn"),
    evaluations_per_trial: int = 64,
) -> dict:
    """Run RNN using the parent pipeline and CatBoost-style evaluation stages."""

    preprocessed = experiment.load_and_preprocess()
    modelling_ready = experiment.extract_features(preprocessed)
    X_initial = modelling_ready.initial_training
    X_validate = modelling_ready.validation
    y_train = preprocessed.initial_training["home_victory"].to_list()
    y_validate = preprocessed.validation["home_victory"].to_list()

    selector = FeatureSelector(
        X_initial, X_validate, y_train, y_validate, metric="calibration"
    )
    selected_features = selector.run_feature_selection(save_results=False)
    X_initial = X_initial[selected_features]
    X_validate = X_validate[selected_features]

    hpo = RNNCalibrationHPO(
        X_initial,
        X_validate,
        y_train,
        y_validate,
        num_trials=experiment.num_hpo_trials,
        num_scoring_runs=experiment.num_prediction_runs,
        evaluations_per_trial=evaluations_per_trial,
    )
    best = hpo.run_hpo()

    test_result = evaluate_rnn(
        modelling_ready.extended_training[selected_features],
        modelling_ready.test[selected_features],
        preprocessed.extended_training["home_victory"].to_list(),
        preprocessed.test["home_victory"].to_list(),
        num_runs=experiment.num_prediction_runs,
        **best,
    )

    final_X, final_y, _ = build_row_sequences(
        modelling_ready.final_training[selected_features],
        preprocessed.final_training["home_victory"].to_list(),
        int(best["sequence_length"]),
    )
    betting_X, betting_y, betting_metadata = build_row_sequences(
        modelling_ready.betting_simulation[selected_features],
        preprocessed.betting_simulation["home_victory"].to_list(),
        int(best["sequence_length"]),
    )
    odds = preprocessed.odds.iloc[
        betting_metadata["source_position"].to_numpy()
    ].reset_index(drop=True)
    simulation = RNNBettingSimulation(
        model_input_size=final_X.shape[-1],
        model_hyperparameters={
            key: best[key] for key in ("hidden_size", "num_layers", "dropout")
        },
        training_hyperparameters={
            key: best[key] for key in ("learning_rate", "batch_size", "epochs")
        },
        X_train_final=final_X,
        y_train_final=final_y.tolist(),
        X_betting_simulation=betting_X,
        y_betting_simulation=betting_y.tolist(),
        odds=odds,
        output_dir=output_dir,
        num_runs=experiment.num_prediction_runs,
    )
    simulation.generate_predictions()
    betting_metrics = simulation.record_metrics()
    fixed_bankroll, fixed_value_bets = simulation.run_betting_simulation("fixed")
    kelly_bankroll, kelly_value_bets = simulation.run_betting_simulation("kelly")

    result = {
        "test_metrics": test_result,
        "betting_simulation_metrics": betting_metrics,
        "features": selected_features,
        "hyperparameters": best,
        "fixed_betting_final_balance": fixed_bankroll[-1],
        "fixed_betting_roi": round(
            100
            * (fixed_bankroll[-1] - simulation.INITIAL_BANKROLL)
            / simulation.INITIAL_BANKROLL,
            2,
        ),
        "kelly_betting_final_balance": kelly_bankroll[-1],
        "kelly_betting_roi": round(
            100
            * (kelly_bankroll[-1] - simulation.INITIAL_BANKROLL)
            / simulation.INITIAL_BANKROLL,
            2,
        ),
        "fixed_betting_value_bets": fixed_value_bets,
        "kelly_betting_value_bets": kelly_value_bets,
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "calibration_driven_rnn_results.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=4)
    return result