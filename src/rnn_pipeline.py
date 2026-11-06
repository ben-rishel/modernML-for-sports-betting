"""GRU-based RNN experiment using the parent pipeline's data splits."""

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from calibra.errors import classwise_ece
from calibra.utils import bin_probabilities, get_classwise_bin_weights
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from rnn_model import RNNClassifier
from rnn_simulate import RNNBettingSimulation
from rnn_training import RNNTrainer
from sequential_data import SequentialDataBuilder, SequentialDataset

CANDIDATE_SEQ_LENGTHS = [5, 10, 15, 20]


def _load_raw_games(years: List[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        path = os.path.join(".", "data", "input", f"results_and_box_scores_{year}.csv")
        df = pd.read_csv(path)
        df["_season"] = year
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _detect_raw_features(games: pd.DataFrame) -> List[str]:
    """Return feature names that have both home_{feat} and away_{feat} columns
    and are fully numeric with no missing values."""
    cols = set(games.columns)
    features = []
    for col in games.columns:
        if col.startswith("home_"):
            feat = col[5:]
            away_col = f"away_{feat}"
            if away_col not in cols:
                continue
            home_numeric = pd.to_numeric(games[col], errors="coerce")
            away_numeric = pd.to_numeric(games[away_col], errors="coerce")
            if home_numeric.isna().any() or away_numeric.isna().any():
                continue
            features.append(feat)
    return sorted(features)


def _build_scaled_dataset(
    raw_df: pd.DataFrame,
    feature_columns: List[str],
    seq_len: int,
    scaler: StandardScaler,
) -> SequentialDataset:
    ds = SequentialDataBuilder(feature_columns, sequence_length=seq_len).build(raw_df)
    num_features = ds.X.shape[-1]
    X_scaled = scaler.transform(ds.X.reshape(-1, num_features)).reshape(ds.X.shape).astype(np.float32)
    return SequentialDataset(X=X_scaled, y=ds.y, metadata=ds.metadata)


class RNNCalibrationHPO:
    """Tune GRU parameters using the same validation protocol as CatBoost."""

    def __init__(
        self,
        seq_cache: Dict[int, Dict[str, SequentialDataset]],
        num_trials: int = 10,
        num_scoring_runs: int = 5,
        evaluations_per_trial: int = 64,
    ):
        self.seq_cache = seq_cache
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
            "hidden_size": hp.choice("hidden_size", [32, 64, 128]),
            "num_layers": hp.choice("num_layers", [1, 2]),
            "dropout": hp.choice("dropout", [0.0, 0.1, 0.2]),
            "learning_rate": hp.loguniform(
                "learning_rate", np.log(0.0001), np.log(0.01)
            ),
            "batch_size": hp.choice("batch_size", [16, 32, 64]),
            "epochs": hp.choice("epochs", [20, 50, 100]),
            "sequence_length": hp.choice("sequence_length", CANDIDATE_SEQ_LENGTHS),
        }

    @staticmethod
    def _decode(best: Dict[str, object]) -> Dict[str, object]:
        num_layers = [1, 2][best["num_layers"]]
        raw_dropout = [0.0, 0.1, 0.2][best["dropout"]]
        return {
            "hidden_size": [32, 64, 128][best["hidden_size"]],
            "num_layers": num_layers,
            "dropout": 0.0 if num_layers == 1 else raw_dropout,
            "learning_rate": float(best["learning_rate"]),
            "batch_size": [16, 32, 64][best["batch_size"]],
            "epochs": [20, 50, 100][best["epochs"]],
            "sequence_length": CANDIDATE_SEQ_LENGTHS[best["sequence_length"]],
        }

    def _fit_and_score(self, params: Dict[str, object], seed: int) -> float:
        seq_len = int(params["sequence_length"])
        split = self.seq_cache[seq_len]
        X_train = split["train"].X
        y_train = split["train"].y
        X_validate = split["validate"].X
        y_validate = split["validate"].y

        num_layers = int(params["num_layers"])
        raw_dropout = float(params.get("dropout", 0.0))
        dropout = 0.0 if num_layers == 1 else raw_dropout

        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        trainer = RNNTrainer(
            RNNClassifier(
                input_size=X_train.shape[-1],
                hidden_size=int(params["hidden_size"]),
                num_layers=num_layers,
                dropout=dropout,
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
                "loss": self._fit_and_score(params, trial_index),
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
    X_train: np.ndarray,
    X_validate: np.ndarray,
    y_train,
    y_validate,
    num_runs: int = 1,
    **parameters,
) -> dict:
    """Average seeded GRU predictions using the CatBoost calibration protocol.

    Expects pre-built, pre-scaled sequence arrays — no internal sequence
    building happens here.
    """
    y_train = np.asarray(y_train, dtype=np.int64)
    y_validate = np.asarray(y_validate, dtype=np.int64)

    probability_sum = np.zeros((len(y_validate), 2), dtype=float)
    calibration_scores = []
    raw_scores = []
    populated_bins = []
    accuracies = []

    num_layers = int(parameters["num_layers"])
    raw_dropout = float(parameters.get("dropout", 0.0))
    dropout = 0.0 if num_layers == 1 else raw_dropout

    for seed in range(num_runs):
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        trainer = RNNTrainer(
            RNNClassifier(
                input_size=X_train.shape[-1],
                hidden_size=int(parameters["hidden_size"]),
                num_layers=num_layers,
                dropout=dropout,
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

    average_probabilities = probability_sum / num_runs
    averaged_calibration_score = RNNCalibrationHPO.calculate_calibration_score(
        y_validate, average_probabilities
    )
    return {
        "calibration_score": averaged_calibration_score,
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
    """Run GRU using the parent pipeline's data splits and CatBoost-style evaluation."""

    preprocessed = experiment.load_and_preprocess()

    all_years = (
        experiment.initial_training_years
        + experiment.validation_years
        + experiment.test_years
        + experiment.betting_simulation_years
    )
    raw_initial  = _load_raw_games(experiment.initial_training_years)
    raw_validate = _load_raw_games(experiment.validation_years)
    raw_extended = _load_raw_games(experiment.initial_training_years + experiment.validation_years)
    raw_test     = _load_raw_games(experiment.test_years)
    raw_final    = _load_raw_games(
        experiment.initial_training_years + experiment.validation_years + experiment.test_years
    )
    raw_betting  = _load_raw_games(experiment.betting_simulation_years)

    feature_columns = _detect_raw_features(raw_initial)

    # Fit scaler on initial training sequences (flattened timesteps)
    _seed_ds = SequentialDataBuilder(feature_columns, sequence_length=5).build(raw_initial)
    scaler = StandardScaler()
    scaler.fit(_seed_ds.X.reshape(-1, _seed_ds.X.shape[-1]))

    # Pre-build scaled sequence caches for all HPO-candidate sequence lengths
    seq_cache = {
        seq_len: {
            "train":    _build_scaled_dataset(raw_initial,  feature_columns, seq_len, scaler),
            "validate": _build_scaled_dataset(raw_validate, feature_columns, seq_len, scaler),
        }
        for seq_len in CANDIDATE_SEQ_LENGTHS
    }

    hpo = RNNCalibrationHPO(
        seq_cache=seq_cache,
        num_trials=experiment.num_hpo_trials,
        num_scoring_runs=experiment.num_prediction_runs,
        evaluations_per_trial=evaluations_per_trial,
    )
    best = hpo.run_hpo()
    seq_len = int(best["sequence_length"])

    extended_ds = _build_scaled_dataset(raw_extended, feature_columns, seq_len, scaler)
    test_ds     = _build_scaled_dataset(raw_test,     feature_columns, seq_len, scaler)
    final_ds    = _build_scaled_dataset(raw_final,    feature_columns, seq_len, scaler)
    betting_ds  = _build_scaled_dataset(raw_betting,  feature_columns, seq_len, scaler)

    test_result = evaluate_rnn(
        extended_ds.X,
        test_ds.X,
        extended_ds.y,
        test_ds.y,
        num_runs=experiment.num_prediction_runs,
        **best,
    )

    # Align odds to betting sequences via Date + Home/Neutral join
    betting_lookup = {
        (str(r["Date"]), str(r["Home/Neutral"])): pos
        for pos, r in enumerate(preprocessed.betting_simulation.to_dict("records"))
    }
    odds_positions = []
    valid_mask = []
    for _, meta_row in betting_ds.metadata.iterrows():
        key = (str(meta_row["Date"]), str(meta_row["Home/Neutral"]))
        if key in betting_lookup:
            odds_positions.append(betting_lookup[key])
            valid_mask.append(True)
        else:
            valid_mask.append(False)
    valid_mask = np.array(valid_mask)
    betting_X = betting_ds.X[valid_mask]
    betting_y = betting_ds.y[valid_mask].tolist()
    odds = preprocessed.odds.iloc[odds_positions].reset_index(drop=True)

    simulation = RNNBettingSimulation(
        model_input_size=final_ds.X.shape[-1],
        model_hyperparameters={
            key: best[key] for key in ("hidden_size", "num_layers", "dropout")
        },
        training_hyperparameters={
            key: best[key] for key in ("learning_rate", "batch_size", "epochs")
        },
        X_train_final=final_ds.X,
        y_train_final=final_ds.y.tolist(),
        X_betting_simulation=betting_X,
        y_betting_simulation=betting_y,
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
        "features": feature_columns,
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
