import json
import sys
from types import ModuleType

import numpy as np
import pandas as pd

from catboost_simulate import CatBoostBettingSimulation


def make_simulation(output_dir, num_runs=1):
    X_train = pd.DataFrame({"feature": [0.0, 1.0, 0.2, 0.8]})
    X_betting = pd.DataFrame({"feature": [0.1, 0.9]})
    odds = pd.DataFrame({"home_odds": [2.0, 2.0], "away_odds": [2.0, 2.0]})

    return CatBoostBettingSimulation(
        optimal_features=["feature"],
        optimal_hyperparameters={"depth": 4, "iterations": 2, "learning_rate": 0.05},
        X_train_final=X_train,
        y_train_final=[0, 1, 0, 1],
        X_betting_simulation=X_betting,
        y_betting_simulation=[1, 0],
        odds=odds,
        output_dir=str(output_dir),
        num_runs=num_runs,
    )


def test_catboost_train_model_fits_selected_features(monkeypatch, tmp_path):
    captured = {}

    class FakeCatBoostClassifier:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit(self, X, y):
            captured["columns"] = list(X.columns)
            captured["y"] = y
            return self

    fake_catboost = ModuleType("catboost")
    fake_catboost.CatBoostClassifier = FakeCatBoostClassifier
    monkeypatch.setitem(sys.modules, "catboost", fake_catboost)

    simulation = make_simulation(tmp_path)
    model = simulation.train_model(random_state=7)

    assert isinstance(model, FakeCatBoostClassifier)
    assert captured["columns"] == ["feature"]
    assert captured["y"] == [0, 1, 0, 1]
    assert captured["kwargs"]["random_seed"] == 7
    assert captured["kwargs"]["depth"] == 4


def test_fixed_betting_uses_parent_rules_and_saves_catboost_outputs(tmp_path):
    simulation = make_simulation(tmp_path)
    simulation.prediction_array = np.asarray([[0.1, 0.9], [0.8, 0.2]])

    bankroll, value_bets = simulation.run_betting_simulation("fixed")

    assert bankroll == [10_000, 10_100, 10_200]
    assert value_bets["predicted_probability"] == [0.9, 0.8]
    assert value_bets["implied_probability"] == [0.5, 0.5]

    with open(tmp_path / "fixed_betting_results.json", encoding="utf-8") as file:
        results = json.load(file)
    with open(tmp_path / "fixed_betting_summary.json", encoding="utf-8") as file:
        summary = json.load(file)

    assert results["final_balance"] == 10_200
    assert results["ROI"] == 2.0
    assert summary["bets_made"] == 2
    assert summary["bets_won"] == 2


def test_record_metrics_writes_to_catboost_output_directory(tmp_path):
    simulation = make_simulation(tmp_path)
    simulation.prediction_array = np.asarray([[0.1, 0.9], [0.8, 0.2]])

    metrics = simulation.record_metrics()

    with open(tmp_path / "catboost_betting_simulation_metrics.json", encoding="utf-8") as file:
        saved_metrics = json.load(file)

    assert metrics == saved_metrics
    assert saved_metrics["accuracy"] == 1.0
    assert (tmp_path / "catboost_betting_simulation_metrics.json").exists()