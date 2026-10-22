import json
import os
from typing import Dict, List

import numpy as np
from calibra.errors import classwise_ece
from calibra.utils import bin_probabilities, get_classwise_bin_weights
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from feature_selection import FeatureSelector
from sklearn.metrics import accuracy_score
from catboost_simulate import CatBoostBettingSimulation


class CatBoostCalibrationHPO:
    """Validation-based HPO side-branch for CatBoost calibration benchmarking.

    This mirrors the existing project protocol: tune on validation data using the
    same calibration objective, then keep the test split untouched. This does not
    alter the main workflow in the repo.
    """

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
        if self.num_trials < 1 or self.num_scoring_runs < 1 or self.evaluations_per_trial < 1:
            raise ValueError("HPO trials, scoring runs, and evaluations per trial must be positive.")

    @staticmethod
    def calculate_calibration_score(
        y_true: List[int],
        y_pred_proba,
        num_bins: int = 20,
    ) -> float:
        y_true = np.asarray(y_true, dtype=int)
        y_pred_proba = np.asarray(y_pred_proba, dtype=float)

        if y_pred_proba.ndim != 2 or y_pred_proba.shape[1] != 2:
            raise ValueError("y_pred_proba must be shaped like (n_samples, 2).")
        if y_true.shape[0] != y_pred_proba.shape[0]:
            raise ValueError("y_true and y_pred_proba must contain the same number of samples.")

        binned_probabilities = bin_probabilities(y_pred_proba, y_true, num_bins=num_bins)
        bin_weights = get_classwise_bin_weights(binned_probabilities)[0]
        non_empty_bins = [1 for weight in bin_weights if weight > 0]

        if sum(non_empty_bins) >= 0.8 * num_bins:
            return float(classwise_ece(y_pred_proba, y_true, num_bins))
        return 1.0

    def get_score(self, model) -> float:
        """Score the model under the project’s calibration criterion."""
        predictions = model.predict_proba(self.X_validate)
        return self.calculate_calibration_score(self.y_validate, predictions)

    def define_search_space(self) -> Dict[str, object]:
        """Define a Hyperopt search space matching the repo's Bayesian HPO setup."""
        return {
            "depth": hp.choice("depth", [4, 6, 8, 10]),
            "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
            "iterations": hp.choice("iterations", [200, 500, 1000]),
        }

    def objective(self, params: Dict[str, object], random_seed: int) -> Dict[str, float]:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CatBoost is not installed. Install catboost before running the side-branch benchmark."
            ) from exc

        model = CatBoostClassifier(
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            iterations=params["iterations"],
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            random_seed=random_seed,
        )
        model.fit(self.X_train, self.y_train)
        return {"loss": self.get_score(model), "status": STATUS_OK}

    def decode_hyperparameters(self, best: Dict[str, object]) -> Dict[str, float]:
        return {
            "depth": [4, 6, 8, 10][best["depth"]],
            "learning_rate": float(best["learning_rate"]),
            "iterations": [200, 500, 1000][best["iterations"]],
        }

    def score_hyperparameters(
        self,
        params: Dict[str, float],
    ) -> float:
        """Average the validation score over the configured random seeds."""
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CatBoost is not installed. Install catboost before running the side-branch benchmark."
            ) from exc

        cumulative_score = 0.0
        for random_seed in range(self.num_scoring_runs):
            model = CatBoostClassifier(
                **params,
                loss_function="Logloss",
                verbose=False,
                allow_writing_files=False,
                random_seed=random_seed,
            )
            model.fit(self.X_train, self.y_train)
            cumulative_score += self.get_score(model)
        return cumulative_score / self.num_scoring_runs

    def run_single_hpo_trial(self, trial_index: int) -> Dict[str, object]:
        """Run one seeded TPE search, then score its best parameters repeatedly."""
        trials = Trials()
        best = fmin(
            fn=lambda params: self.objective(params, random_seed=trial_index),
            space=self.define_search_space(),
            algo=tpe.suggest,
            max_evals=self.evaluations_per_trial,
            trials=trials,
            rstate=np.random.default_rng(trial_index),
        )
        best_hyperparameters = self.decode_hyperparameters(best)
        average_score = self.score_hyperparameters(best_hyperparameters)
        return {"hyperparameters": best_hyperparameters, "score": average_score}

    def run_hpo(self) -> Dict[str, float]:
        """Run independent TPE searches and select the best repeated validation score."""
        self.hpo_results = [
            self.run_single_hpo_trial(trial_index)
            for trial_index in range(self.num_trials)
        ]
        return min(self.hpo_results, key=lambda result: result["score"])["hyperparameters"]


def calculate_calibration_score(
    y_true: List[int],
    y_pred_proba,
    num_bins: int = 20,
) -> float:
    """Public helper that mirrors the repo's calibration scoring logic."""
    return CatBoostCalibrationHPO.calculate_calibration_score(y_true, y_pred_proba, num_bins=num_bins)


def evaluate_catboost_calibration(
    X_train,
    X_validate,
    y_train: List[int],
    y_validate: List[int],
    num_runs: int = 1,
    **model_kwargs,
) -> dict:
    """Average seeded CatBoost predictions and score them on the supplied split."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CatBoost is not installed. Add it to the environment before running this side-branch benchmark."
        ) from exc

    if num_runs < 1:
        raise ValueError("num_runs must be at least 1.")

    probability_sum = np.zeros((len(X_validate), 2), dtype=float)
    calibration_scores = []
    raw_calibration_scores = []
    populated_bins_per_run = []
    accuracies = []
    model = None
    for random_seed in range(num_runs):
        model = CatBoostClassifier(
            loss_function="Logloss",
            verbose=False,
            allow_writing_files=False,
            random_seed=random_seed,
            **model_kwargs,
        )
        model.fit(X_train, y_train)
        predicted_probabilities = np.asarray(model.predict_proba(X_validate))
        probability_sum += predicted_probabilities
        binned_probabilities = bin_probabilities(
            predicted_probabilities, y_validate, num_bins=20
        )
        bin_weights = get_classwise_bin_weights(binned_probabilities)[0]
        populated_bins = sum(float(weight) > 0 for weight in bin_weights)
        populated_bins_per_run.append(populated_bins)
        raw_calibration_scores.append(
            float(classwise_ece(predicted_probabilities, y_validate, 20))
        )
        calibration_scores.append(
            calculate_calibration_score(y_validate, predicted_probabilities)
        )
        predicted_labels = (predicted_probabilities[:, 1] >= 0.5).astype(int)
        accuracies.append(accuracy_score(y_validate, predicted_labels))

    average_probabilities = probability_sum / num_runs

    return {
        "calibration_score": float(np.mean(calibration_scores)),
        "raw_calibration_score": float(np.mean(raw_calibration_scores)),
        "calibration_scores_per_run": [float(score) for score in calibration_scores],
        "raw_calibration_scores_per_run": raw_calibration_scores,
        "populated_bins_per_run": populated_bins_per_run,
        "calibration_guard_failures": sum(
            populated_bins < 0.8 * 20 for populated_bins in populated_bins_per_run
        ),
        "accuracy": float(np.mean(accuracies)),
        "probabilities": average_probabilities.tolist(),
        "model": model,
    }


def run_catboost_calibration_side_experiment(
    experiment,
    metric: str = "calibration",
    output_dir: str = os.path.join(".", "data", "output", "catboost"),
) -> dict:
    """Run a CatBoost calibration benchmark using the same split logic as the repo.

    This intentionally does not modify the existing accuracy/calibration selection
    workflow. It consumes the prepared datasets and evaluates CatBoost under the
    same calibration criterion used elsewhere in the project.
    """
    preprocessed_datasets = experiment.load_and_preprocess()
    modelling_ready_datasets = experiment.extract_features(preprocessed_datasets)

    X_initial = modelling_ready_datasets.initial_training
    X_validate = modelling_ready_datasets.validation
    y_train = preprocessed_datasets.initial_training["home_victory"].to_list()
    y_validate = preprocessed_datasets.validation["home_victory"].to_list()

    feature_selector = FeatureSelector(
        X_initial,
        X_validate,
        y_train,
        y_validate,
        metric="calibration",
    )
    selected_features = feature_selector.run_feature_selection(save_results=False)
    X_initial = X_initial[selected_features]
    X_validate = X_validate[selected_features]

    hpo = CatBoostCalibrationHPO(
        X_initial,
        X_validate,
        y_train,
        y_validate,
        num_trials=experiment.num_hpo_trials,
        num_scoring_runs=experiment.num_prediction_runs,
    )
    best_hyperparameters = hpo.run_hpo()

    X_extended_training = modelling_ready_datasets.extended_training[selected_features]
    X_test = modelling_ready_datasets.test[selected_features]
    y_extended_training = preprocessed_datasets.extended_training["home_victory"].to_list()
    y_test = preprocessed_datasets.test["home_victory"].to_list()
    test_result = evaluate_catboost_calibration(
        X_extended_training,
        X_test,
        y_extended_training,
        y_test,
        num_runs=experiment.num_prediction_runs,
        **best_hyperparameters,
    )

    X_final_training = modelling_ready_datasets.final_training[selected_features]
    X_betting = modelling_ready_datasets.betting_simulation[selected_features]
    y_final_training = preprocessed_datasets.final_training["home_victory"].to_list()
    y_betting = preprocessed_datasets.betting_simulation["home_victory"].to_list()
    simulation = CatBoostBettingSimulation(
        optimal_features=selected_features,
        optimal_hyperparameters=best_hyperparameters,
        X_train_final=X_final_training,
        y_train_final=y_final_training,
        X_betting_simulation=X_betting,
        y_betting_simulation=y_betting,
        odds=preprocessed_datasets.odds,
        output_dir=output_dir,
        num_runs=experiment.num_prediction_runs,
    )
    simulation.generate_predictions()
    betting_simulation_metrics = simulation.record_metrics()
    fixed_bankroll, fixed_value_bets = simulation.run_betting_simulation("fixed")
    kelly_bankroll, kelly_value_bets = simulation.run_betting_simulation("kelly")

    os.makedirs(output_dir, exist_ok=True)
    fixed_betting_roi = round(
        100 * (fixed_bankroll[-1] - simulation.INITIAL_BANKROLL) / simulation.INITIAL_BANKROLL,
        2,
    )
    kelly_betting_roi = round(
        100 * (kelly_bankroll[-1] - simulation.INITIAL_BANKROLL) / simulation.INITIAL_BANKROLL,
        2,
    )
    with open(os.path.join(output_dir, f"{metric}_driven_catboost_results.json"), "w") as file:
        json.dump(
            {
                "test_calibration_score": test_result["calibration_score"],
                "test_raw_calibration_score": test_result["raw_calibration_score"],
                "test_calibration_scores_per_run": test_result["calibration_scores_per_run"],
                "test_raw_calibration_scores_per_run": test_result["raw_calibration_scores_per_run"],
                "test_populated_bins_per_run": test_result["populated_bins_per_run"],
                "test_calibration_guard_failures": test_result["calibration_guard_failures"],
                "test_accuracy": test_result["accuracy"],
                "betting_simulation_calibration_score": betting_simulation_metrics["calibration_score"],
                "betting_simulation_accuracy": betting_simulation_metrics["accuracy"],
                "features": selected_features,
                "hyperparameters": best_hyperparameters,
                "fixed_betting_final_balance": fixed_bankroll[-1],
                "fixed_betting_roi": fixed_betting_roi,
                "kelly_betting_final_balance": kelly_bankroll[-1],
                "kelly_betting_roi": kelly_betting_roi,
            },
            file,
            indent=4,
        )
    with open(os.path.join(output_dir, "catboost_betting_simulation_data.json"), "w") as file:
        json.dump(
            {
                "fixed_betting_bankroll": fixed_bankroll,
                "fixed_betting_value_bets": fixed_value_bets,
                "kelly_betting_bankroll": kelly_bankroll,
                "kelly_betting_value_bets": kelly_value_bets,
            },
            file,
            indent=4,
        )

    return {
        "test_metrics": {
            "calibration_score": test_result["calibration_score"],
            "raw_calibration_score": test_result["raw_calibration_score"],
            "calibration_scores_per_run": test_result["calibration_scores_per_run"],
            "raw_calibration_scores_per_run": test_result["raw_calibration_scores_per_run"],
            "populated_bins_per_run": test_result["populated_bins_per_run"],
            "calibration_guard_failures": test_result["calibration_guard_failures"],
            "accuracy": test_result["accuracy"],
        },
        "betting_simulation_metrics": betting_simulation_metrics,
        "features": selected_features,
        "hyperparameters": best_hyperparameters,
        "fixed_betting_bankroll": fixed_bankroll,
        "fixed_betting_value_bets": fixed_value_bets,
        "fixed_betting_roi": fixed_betting_roi,
        "kelly_betting_bankroll": kelly_bankroll,
        "kelly_betting_value_bets": kelly_value_bets,
        "kelly_betting_roi": kelly_betting_roi,
        "n_test_examples": len(y_test),
        "n_betting_examples": len(y_betting),
    }
