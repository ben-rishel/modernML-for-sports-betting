import json

import numpy as np
import pandas as pd

from src.rnn_simulate import RNNBettingSimulation


def make_simulation(output_dir):
    X_train = np.array(
        [[[0.0], [0.1]], [[1.0], [0.9]], [[0.2], [0.1]], [[0.8], [0.9]]],
        dtype=np.float32,
    )
    X_betting = np.array([[[0.1], [0.2]], [[0.9], [0.8]]], dtype=np.float32)
    return RNNBettingSimulation(
        model_input_size=1,
        model_hyperparameters={"hidden_size": 4, "num_layers": 1, "dropout": 0.0},
        training_hyperparameters={
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 2,
        },
        X_train_final=X_train,
        y_train_final=[0, 1, 0, 1],
        X_betting_simulation=X_betting,
        y_betting_simulation=[1, 0],
        odds=pd.DataFrame({"home_odds": [2.0, 2.0], "away_odds": [2.0, 2.0]}),
        output_dir=str(output_dir),
    )


def test_rnn_simulation_generates_predictions(tmp_path):
    simulation = make_simulation(tmp_path)
    simulation.generate_predictions()

    assert simulation.prediction_array.shape == (2, 2)
    np.testing.assert_allclose(simulation.prediction_array.sum(axis=1), [1.0, 1.0])


def test_rnn_simulation_uses_parent_betting_rules(tmp_path):
    simulation = make_simulation(tmp_path)
    simulation.prediction_array = np.asarray([[0.1, 0.9], [0.8, 0.2]])

    bankroll, value_bets = simulation.run_betting_simulation("fixed")

    assert bankroll == [10_000, 10_100, 10_200]
    assert value_bets["predicted_probability"] == [0.9, 0.8]


def test_rnn_simulation_records_metrics(tmp_path):
    simulation = make_simulation(tmp_path)
    simulation.prediction_array = np.asarray([[0.1, 0.9], [0.8, 0.2]])

    metrics = simulation.record_metrics()

    with open(
        tmp_path / "rnn_betting_simulation_metrics.json", encoding="utf-8"
    ) as file:
        saved_metrics = json.load(file)
    assert metrics == saved_metrics
    assert metrics["accuracy"] == 1.0
