from types import SimpleNamespace

import pandas as pd
import numpy as np

import catboost_calibration
from catboost_calibration import (
    CatBoostCalibrationHPO,
    calculate_calibration_score,
    evaluate_catboost_calibration,
    run_catboost_calibration_side_experiment,
)


def test_calibration_score_is_finite():
    y_true = [0, 0, 1, 1]
    y_score = [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.3, 0.7],
    ]

    score = calculate_calibration_score(y_true, y_score)

    assert isinstance(score, float)
    assert score >= 0


def test_catboost_hpo_returns_hyperparameters():
    X_train = [[0.1, 0.1], [0.2, 0.2], [0.9, 0.9], [1.0, 1.0]]
    X_validate = [[0.15, 0.15], [0.85, 0.85]]
    y_train = [0, 0, 1, 1]
    y_validate = [0, 1]

    hpo = CatBoostCalibrationHPO(
        X_train,
        X_validate,
        y_train,
        y_validate,
        num_trials=2,
        num_scoring_runs=1,
        evaluations_per_trial=2,
    )

    params = hpo.run_hpo()

    assert isinstance(params, dict)
    assert {"depth", "learning_rate", "iterations"}.issubset(set(params.keys()))


def test_catboost_hpo_matches_parent_search_restarts(monkeypatch):
    hpo = CatBoostCalibrationHPO(
        [[0.0], [1.0]],
        [[0.2], [0.8]],
        [0, 1],
        [0, 1],
        num_trials=3,
        num_scoring_runs=2,
    )
    calls = []

    def fake_fmin(fn, space, algo, max_evals, trials, rstate):
        calls.append(max_evals)
        trial_index = len(calls) - 1
        return {
            "depth": trial_index,
            "learning_rate": 0.02 + trial_index * 0.01,
            "iterations": trial_index,
        }

    scores_by_depth = {4: 0.3, 6: 0.1, 8: 0.2}
    monkeypatch.setattr(catboost_calibration, "fmin", fake_fmin)
    monkeypatch.setattr(
        hpo,
        "score_hyperparameters",
        lambda params: scores_by_depth[params["depth"]],
    )

    best = hpo.run_hpo()

    assert calls == [64, 64, 64]
    assert best["depth"] == 6
    assert [trial["score"] for trial in hpo.hpo_results] == [0.3, 0.1, 0.2]


def test_test_evaluation_averages_seeded_predictions(monkeypatch):
    captured_seeds = []

    class FakeCatBoostClassifier:
        def __init__(self, random_seed, **kwargs):
            self.random_seed = random_seed
            captured_seeds.append(random_seed)

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            predictions = {
                0: [[0.49, 0.51], [0.1, 0.9]],
                1: [[0.51, 0.49], [0.1, 0.9]],
                2: [[0.51, 0.49], [0.1, 0.9]],
            }
            return predictions[self.random_seed]

    fake_catboost = SimpleNamespace(CatBoostClassifier=FakeCatBoostClassifier)
    monkeypatch.setitem(__import__("sys").modules, "catboost", fake_catboost)

    result = evaluate_catboost_calibration(
        pd.DataFrame({"feature": [0, 1]}),
        pd.DataFrame({"feature": [0, 1]}),
        [0, 1],
        [0, 1],
        num_runs=3,
        depth=4,
        iterations=2,
        learning_rate=0.05,
    )

    assert captured_seeds == [0, 1, 2]
    np.testing.assert_allclose(
        result["probabilities"], [[0.5033333333, 0.4966666667], [0.1, 0.9]]
    )
    assert result["accuracy"] == 0.8333333333333334
    assert result["calibration_score"] == 1.0
    assert result["raw_calibration_score"] < 1.0
    assert result["calibration_guard_failures"] == 3
    assert result["populated_bins_per_run"] == [2, 2, 2]


def test_side_experiment_uses_calibration_selected_features(monkeypatch, tmp_path):
    selected_features = ["calibration_feature"]
    captured = {}

    class FakeFeatureSelector:
        def __init__(self, X_train, X_validate, y_train, y_validate, metric):
            captured["selector_metric"] = metric

        def run_feature_selection(self, save_results=True):
            captured["save_results"] = save_results
            return selected_features

    class FakeHPO:
        def __init__(self, X_train, X_validate, y_train, y_validate, **kwargs):
            captured["hpo_train_columns"] = list(X_train.columns)
            captured["hpo_validate_columns"] = list(X_validate.columns)
            captured["hpo_train_rows"] = len(X_train)
            captured["hpo_validate_rows"] = len(X_validate)

        def run_hpo(self):
            return {"depth": 4, "learning_rate": 0.05, "iterations": 200}

    def fake_evaluate(X_train, X_validate, y_train, y_validate, **model_kwargs):
        captured["eval_train_columns"] = list(X_train.columns)
        captured["eval_validate_columns"] = list(X_validate.columns)
        captured["eval_train_rows"] = len(X_train)
        captured["eval_test_rows"] = len(X_validate)
        captured["eval_num_runs"] = model_kwargs["num_runs"]
        return {
            "calibration_score": 0.1,
            "raw_calibration_score": 0.05,
            "calibration_scores_per_run": [0.1],
            "raw_calibration_scores_per_run": [0.05],
            "populated_bins_per_run": [20],
            "calibration_guard_failures": 0,
            "accuracy": 0.75,
        }

    class FakeSimulation:
        INITIAL_BANKROLL = 10_000

        def __init__(self, **kwargs):
            captured["simulation_kwargs"] = kwargs

        def generate_predictions(self):
            captured["predictions_generated"] = True

        def record_metrics(self):
            return {"calibration_score": 0.08, "accuracy": 0.7}

        def run_betting_simulation(self, rule):
            captured.setdefault("betting_rules", []).append(rule)
            tracker = [10_000, 10_100] if rule == "fixed" else [10_000, 9_900]
            return tracker, {"predicted_probability": [], "implied_probability": []}

    class FakeExperiment:
        num_hpo_trials = 2
        num_prediction_runs = 1

        def load_and_preprocess(self):
            return SimpleNamespace(
                initial_training=pd.DataFrame({"home_victory": [0, 1]}),
                validation=pd.DataFrame({"home_victory": [0, 1]}),
                extended_training=pd.DataFrame({"home_victory": [0, 1, 1, 0]}),
                test=pd.DataFrame({"home_victory": [1, 0]}),
                final_training=pd.DataFrame({"home_victory": [0, 1, 1, 0, 1, 0]}),
                betting_simulation=pd.DataFrame({"home_victory": [1, 0]}),
                odds=pd.DataFrame({"home_odds": [2.0, 2.0], "away_odds": [2.0, 2.0]}),
            )

        def extract_features(self, _):
            def frame(rows):
                return pd.DataFrame(
                    {
                        "calibration_feature": list(range(rows)),
                        "other_feature": list(range(rows, 2 * rows)),
                    }
                )

            return SimpleNamespace(**{
                "initial_training": frame(2),
                "validation": frame(2),
                "extended_training": frame(4),
                "test": frame(2),
                "final_training": frame(6),
                "betting_simulation": frame(2),
            })

    monkeypatch.setattr(catboost_calibration, "FeatureSelector", FakeFeatureSelector)
    monkeypatch.setattr(catboost_calibration, "CatBoostCalibrationHPO", FakeHPO)
    monkeypatch.setattr(catboost_calibration, "evaluate_catboost_calibration", fake_evaluate)
    monkeypatch.setattr(catboost_calibration, "CatBoostBettingSimulation", FakeSimulation)

    result = run_catboost_calibration_side_experiment(FakeExperiment(), output_dir=str(tmp_path))

    assert captured["selector_metric"] == "calibration"
    assert captured["save_results"] is False
    assert captured["hpo_train_columns"] == selected_features
    assert captured["hpo_validate_columns"] == selected_features
    assert captured["eval_train_columns"] == selected_features
    assert captured["eval_validate_columns"] == selected_features
    assert captured["hpo_train_rows"] == 2
    assert captured["hpo_validate_rows"] == 2
    assert captured["eval_train_rows"] == 4
    assert captured["eval_test_rows"] == 2
    assert captured["eval_num_runs"] == 1
    assert captured["simulation_kwargs"]["X_train_final"].shape[0] == 6
    assert captured["simulation_kwargs"]["X_betting_simulation"].shape[0] == 2
    assert list(captured["simulation_kwargs"]["X_train_final"].columns) == selected_features
    assert list(captured["simulation_kwargs"]["X_betting_simulation"].columns) == selected_features
    assert captured["predictions_generated"] is True
    assert captured["betting_rules"] == ["fixed", "kelly"]
    assert result["features"] == selected_features
    assert result["test_metrics"]["accuracy"] == 0.75
    assert result["betting_simulation_metrics"]["accuracy"] == 0.7
    assert result["fixed_betting_roi"] == 1.0
    assert result["kelly_betting_roi"] == -1.0
