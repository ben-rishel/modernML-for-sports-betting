from types import SimpleNamespace

import numpy as np
import pandas as pd

import src.rnn_pipeline as rnn_pipeline
from src.rnn_pipeline import (
    RNNCalibrationHPO,
    evaluate_rnn,
    run_rnn_calibration_side_experiment,
)


def _frames():
    def frame(size):
        return pd.DataFrame(
            {
                "selected": np.arange(size, dtype=float),
                "other": np.arange(size, dtype=float) + 10,
            }
        )

    return frame


def test_rnn_calibration_score_is_finite():
    score = RNNCalibrationHPO.calculate_calibration_score(
        [0, 0, 1, 1],
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.3, 0.7]],
    )

    assert isinstance(score, float)
    assert score >= 0


def test_rnn_hpo_restarts_search_and_selects_lowest_score(monkeypatch):
    hpo = RNNCalibrationHPO(
        pd.DataFrame({"feature": np.arange(20, dtype=float)}),
        pd.DataFrame({"feature": np.arange(20, dtype=float)}),
        [0, 1] * 10,
        [0, 1] * 10,
        num_trials=3,
        num_scoring_runs=2,
        evaluations_per_trial=64,
    )
    calls = []

    def fake_fmin(fn, space, algo, max_evals, trials, rstate):
        calls.append(max_evals)
        return {
            "hidden_size": len(calls) - 1,
            "num_layers": 0,
            "learning_rate": 0.001,
            "batch_size": 1,
            "epochs": 0,
            "sequence_length": 0,
        }

    scores = [0.3, 0.1, 0.2]
    score_calls = []

    def fake_fit_and_score(params, seed):
        score_calls.append((params, seed))
        return scores[len(calls) - 1]

    monkeypatch.setattr(rnn_pipeline, "fmin", fake_fmin)
    monkeypatch.setattr(hpo, "_fit_and_score", fake_fit_and_score)

    best = hpo.run_hpo()

    assert calls == [64, 64, 64]
    assert best["hidden_size"] == 32
    assert [trial["score"] for trial in hpo.hpo_results] == scores
    assert [seed for _, seed in score_calls] == [0, 1, 0, 1, 0, 1]


def test_evaluate_rnn_averages_seeded_predictions(monkeypatch):
    captured_seeds = []

    class FakeTrainer:
        def __init__(self, model, **kwargs):
            captured_seeds.append(model.seed)

        def fit(self, X, y):
            return None

        def predict_proba(self, X):
            if captured_seeds[-1] == 0:
                return np.tile([[0.49, 0.51]], (len(X), 1))
            return np.tile([[0.51, 0.49]], (len(X), 1))

    class FakeModel:
        def __init__(self, **kwargs):
            self.seed = len(captured_seeds)

    monkeypatch.setattr(rnn_pipeline, "RNNTrainer", FakeTrainer)
    monkeypatch.setattr(rnn_pipeline, "RNNClassifier", FakeModel)

    features = pd.DataFrame({"feature": np.arange(6, dtype=float)})
    result = evaluate_rnn(
        features,
        features,
        [0, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 1],
        num_runs=2,
        hidden_size=4,
        num_layers=1,
        dropout=0.0,
        learning_rate=0.001,
        batch_size=2,
        epochs=1,
        sequence_length=2,
    )

    assert captured_seeds == [0, 1]
    np.testing.assert_allclose(
        result["probabilities"],
        np.tile([[0.5, 0.5]], (4, 1)),
    )
    assert result["accuracy"] == 0.5


def test_side_experiment_uses_parent_pipeline_and_selected_features(monkeypatch, tmp_path):
    selected_features = ["selected"]
    frame = _frames()
    captured = {}

    class FakeFeatureSelector:
        def __init__(self, X_train, X_validate, y_train, y_validate, metric):
            captured["selector_metric"] = metric

        def run_feature_selection(self, save_results=True):
            captured["save_results"] = save_results
            return selected_features

    class FakeHPO:
        def __init__(self, X_train, X_validate, y_train, y_validate, **kwargs):
            captured["hpo_columns"] = list(X_train.columns)
            captured["hpo_kwargs"] = kwargs

        def run_hpo(self):
            return {
                "hidden_size": 4,
                "num_layers": 1,
                "dropout": 0.0,
                "learning_rate": 0.001,
                "batch_size": 2,
                "epochs": 1,
                "sequence_length": 2,
            }

    def fake_evaluate(X_train, X_validate, y_train, y_validate, **kwargs):
        captured["evaluation_columns"] = list(X_train.columns)
        captured["evaluation_runs"] = kwargs["num_runs"]
        return {
            "calibration_score": 0.1,
            "raw_calibration_score": 0.05,
            "calibration_scores_per_run": [0.1],
            "raw_calibration_scores_per_run": [0.05],
            "populated_bins_per_run": [20],
            "calibration_guard_failures": 0,
            "accuracy": 0.75,
            "probabilities": [],
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
            captured.setdefault("rules", []).append(rule)
            return [10_000, 10_100 if rule == "fixed" else 9_900], {}

    class FakeExperiment:
        num_hpo_trials = 2
        num_prediction_runs = 3

        def load_and_preprocess(self):
            def labels(size):
                return pd.DataFrame({"home_victory": [0, 1] * (size // 2)})

            return SimpleNamespace(
                initial_training=labels(6),
                validation=labels(6),
                extended_training=labels(8),
                test=labels(6),
                final_training=labels(10),
                betting_simulation=labels(6),
                odds=pd.DataFrame(
                    {"home_odds": [2.0] * 6, "away_odds": [2.0] * 6}
                ),
            )

        def extract_features(self, _):
            return SimpleNamespace(
                initial_training=frame(6),
                validation=frame(6),
                extended_training=frame(8),
                test=frame(6),
                final_training=frame(10),
                betting_simulation=frame(6),
            )

    monkeypatch.setattr(rnn_pipeline, "FeatureSelector", FakeFeatureSelector)
    monkeypatch.setattr(rnn_pipeline, "RNNCalibrationHPO", FakeHPO)
    monkeypatch.setattr(rnn_pipeline, "evaluate_rnn", fake_evaluate)
    monkeypatch.setattr(rnn_pipeline, "RNNBettingSimulation", FakeSimulation)

    result = run_rnn_calibration_side_experiment(
        FakeExperiment(), output_dir=str(tmp_path)
    )

    assert captured["selector_metric"] == "calibration"
    assert captured["save_results"] is False
    assert captured["hpo_columns"] == selected_features
    assert captured["evaluation_columns"] == selected_features
    assert captured["hpo_kwargs"] == {
        "num_trials": 2,
        "num_scoring_runs": 3,
        "evaluations_per_trial": 64,
    }
    assert captured["evaluation_runs"] == 3
    assert captured["predictions_generated"] is True
    assert captured["rules"] == ["fixed", "kelly"]
    assert result["features"] == selected_features
    assert result["fixed_betting_roi"] == 1.0
    assert result["kelly_betting_roi"] == -1.0
    assert (tmp_path / "calibration_driven_rnn_results.json").exists()
