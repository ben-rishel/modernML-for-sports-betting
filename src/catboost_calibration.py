import json
import os
from typing import Dict, List

import numpy as np
from calibra.errors import classwise_ece
from calibra.utils import bin_probabilities, get_classwise_bin_weights
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


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
    ):
        self.X_train = X_train.copy()
        self.X_validate = X_validate.copy()
        self.y_train = list(y_train)
        self.y_validate = list(y_validate)
        self.num_trials = num_trials
        self.num_scoring_runs = num_scoring_runs

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

    def objective(self, params: Dict[str, object]) -> Dict[str, float]:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CatBoost is not installed. Install catboost before running the side-branch benchmark."
            ) from exc

        cumulative_score = 0.0
        for r in range(self.num_scoring_runs):
            model = CatBoostClassifier(
                depth=params["depth"],
                learning_rate=params["learning_rate"],
                iterations=params["iterations"],
                loss_function="Logloss",
                verbose=False,
                allow_writing_files=False,
                random_seed=r,
            )
            model.fit(self.X_train, self.y_train)
            cumulative_score += self.get_score(model)

        average_score = cumulative_score / self.num_scoring_runs
        return {"loss": average_score, "status": STATUS_OK}

    def run_hpo(self) -> Dict[str, float]:
        """Tune CatBoost on validation data using a Bayesian optimisation loop."""
        trials = Trials()
        best = fmin(
            fn=self.objective,
            space=self.define_search_space(),
            algo=tpe.suggest,
            max_evals=self.num_trials,
            trials=trials,
            rstate=np.random.default_rng(0),
        )

        return {
            "depth": [4, 6, 8, 10][best["depth"]],
            "learning_rate": best["learning_rate"],
            "iterations": [200, 500, 1000][best["iterations"]],
        }


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
    **model_kwargs,
) -> dict:
    """Train a CatBoost model using tuned parameters and score it under the repo's calibration metric."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CatBoost is not installed. Add it to the environment before running this side-branch benchmark."
        ) from exc

    model = CatBoostClassifier(
        loss_function="Logloss",
        verbose=False,
        allow_writing_files=False,
        random_seed=0,
        **model_kwargs,
    )
    model.fit(X_train, y_train)

    predicted_probabilities = model.predict_proba(X_validate)
    calibration_score = calculate_calibration_score(y_validate, predicted_probabilities)

    return {
        "calibration_score": float(calibration_score),
        "probabilities": predicted_probabilities.tolist(),
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

    X_train = modelling_ready_datasets.initial_training
    X_validate = modelling_ready_datasets.validation
    y_train = preprocessed_datasets.initial_training["home_victory"].to_list()
    y_validate = preprocessed_datasets.validation["home_victory"].to_list()

    hpo = CatBoostCalibrationHPO(
        X_train,
        X_validate,
        y_train,
        y_validate,
        num_trials=10,
        num_scoring_runs=5,
    )
    best_hyperparameters = hpo.run_hpo()

    result = evaluate_catboost_calibration(
        X_train,
        X_validate,
        y_train,
        y_validate,
        **best_hyperparameters,
    )

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{metric}_driven_catboost_results.json"), "w") as file:
        json.dump({"calibration_score": result["calibration_score"]}, file, indent=4)

    return {
        "calibration_score": result["calibration_score"],
        "hyperparameters": best_hyperparameters,
        "n_validation_examples": len(y_validate),
    }
