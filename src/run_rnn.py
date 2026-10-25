"""Run the standalone RNN calibration and betting experiment."""

import argparse
import json
import os
from typing import Iterable, List, Optional

import pandas as pd

from rnn_calibration import evaluate_rnn_calibration
from rnn_simulate import RNNBettingSimulation
from rnn_model import RNNClassifier
from rnn_training import RNNTrainer
from sequential_data import SequentialDataBuilder
from utils import match_odds_to_games


DEFAULT_FEATURES = ["FG", "FG%", "3P", "3P%", "FT", "FT%"]


def load_games(years: Iterable[int]) -> pd.DataFrame:
    """Load and concatenate raw game data for the supplied seasons."""

    frames = []
    for year in years:
        path = os.path.join(
            ".", "data", "input", f"results_and_box_scores_{year}.csv"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find game data for season {year}: {path}")
        frame = pd.read_csv(path)
        frame["_season"] = year
        frames.append(frame)
    if not frames:
        raise ValueError("At least one season must be supplied.")
    return pd.concat(frames, ignore_index=True)


def build_dataset(
    years: Iterable[int],
    features: List[str],
    sequence_length: int,
):
    """Load seasons and convert them to RNN sequences."""

    games = load_games(years)
    builder = SequentialDataBuilder(
        feature_columns=features,
        sequence_length=sequence_length,
    )
    return builder.build(games)


def run_experiment(args: argparse.Namespace) -> dict:
    """Run training, calibration evaluation, and betting simulation."""

    training = build_dataset(
        args.initial_training_years, args.features, args.sequence_length
    )
    validation = build_dataset(
        args.validation_years, args.features, args.sequence_length
    )
    extended_training = build_dataset(
        args.initial_training_years + args.validation_years,
        args.features,
        args.sequence_length,
    )
    test = build_dataset(args.test_years, args.features, args.sequence_length)
    final_training = build_dataset(
        args.initial_training_years + args.validation_years + args.test_years,
        args.features,
        args.sequence_length,
    )
    betting_games = load_games(args.betting_simulation_years)
    betting_games, matched_odds = match_odds_to_games(
        betting_games, load_odds(args.betting_simulation_years)
    )
    betting = SequentialDataBuilder(
        feature_columns=args.features,
        sequence_length=args.sequence_length,
    ).build(betting_games)
    betting_odds = matched_odds.iloc[
        betting.metadata["source_index"].astype(int).to_numpy()
    ].reset_index(drop=True)

    input_size = training.X.shape[-1]
    model_parameters = {
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    training_parameters = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }

    validation_trainer = RNNTrainer(
        RNNClassifier(input_size=input_size, **model_parameters),
        **training_parameters,
    )
    validation_trainer.fit(
        training.X,
        training.y,
        validation.X,
        validation.y,
    )
    validation_metrics = evaluate_rnn_calibration(
        validation_trainer, validation.X, validation.y
    )

    test_trainer = RNNTrainer(
        RNNClassifier(input_size=input_size, **model_parameters),
        **training_parameters,
    )
    test_trainer.fit(extended_training.X, extended_training.y)
    test_metrics = evaluate_rnn_calibration(test_trainer, test.X, test.y)

    simulation = RNNBettingSimulation(
        model_input_size=input_size,
        model_hyperparameters=model_parameters,
        training_hyperparameters=training_parameters,
        X_train_final=final_training.X,
        y_train_final=final_training.y.tolist(),
        X_betting_simulation=betting.X,
        y_betting_simulation=betting.y.tolist(),
        odds=betting_odds,
        output_dir=args.output_dir,
        num_runs=args.num_runs,
    )
    simulation.generate_predictions()
    betting_metrics = simulation.record_metrics()
    fixed_bankroll, fixed_value_bets = simulation.run_betting_simulation("fixed")
    kelly_bankroll, kelly_value_bets = simulation.run_betting_simulation("kelly")

    result = {
        "features": args.features,
        "sequence_length": args.sequence_length,
        "model_hyperparameters": model_parameters,
        "training_hyperparameters": training_parameters,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "betting_metrics": betting_metrics,
        "fixed_betting_final_balance": fixed_bankroll[-1],
        "kelly_betting_final_balance": kelly_bankroll[-1],
        "fixed_betting_value_bets": fixed_value_bets,
        "kelly_betting_value_bets": kelly_value_bets,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "rnn_results.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=4)
    return result


def load_odds(years: Iterable[int]) -> pd.DataFrame:
    """Load odds for the betting seasons."""

    frames = []
    for year in years:
        path = os.path.join(".", "data", "input", f"nba odds {year - 1}-{str(year)[-2:]}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find odds data for season {year}: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standalone RNN calibration and betting experiment."
    )
    parser.add_argument("--initial-training-years", nargs="+", type=int, default=[2015, 2016])
    parser.add_argument("--validation-years", nargs="+", type=int, default=[2017])
    parser.add_argument("--test-years", nargs="+", type=int, default=[2018])
    parser.add_argument("--betting-simulation-years", nargs="+", type=int, default=[2019])
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--output-dir", default="./data/output/rnn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    print(f"RNN validation accuracy: {result['validation_metrics']['accuracy']:.4f}")
    print(f"RNN test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"RNN betting accuracy: {result['betting_metrics']['accuracy']:.4f}")
    print(f"Fixed betting final balance: {result['fixed_betting_final_balance']:.2f}")
    print(f"Kelly betting final balance: {result['kelly_betting_final_balance']:.2f}")


if __name__ == "__main__":
    main()
