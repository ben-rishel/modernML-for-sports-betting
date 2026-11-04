"""Run the standalone RNN calibration and betting experiment."""

import argparse
import os
from typing import Iterable, List

import pandas as pd

from sequential_data import SequentialDataBuilder


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
    """Run the RNN through the parent pipeline's prepared-data protocol."""

    from pipeline import NBABettingExperiment
    from rnn_pipeline import run_rnn_calibration_side_experiment

    experiment = NBABettingExperiment(
        initial_training_years=args.initial_training_years,
        validation_years=args.validation_years,
        test_years=args.test_years,
        betting_simulation_years=args.betting_simulation_years,
        num_hpo_trials=args.num_hpo_trials,
        num_prediction_runs=args.num_scoring_runs,
    )
    return run_rnn_calibration_side_experiment(
        experiment,
        args.output_dir,
        evaluations_per_trial=args.evaluations_per_trial,
    )


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
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES, help=argparse.SUPPRESS)
    parser.add_argument("--sequence-length", type=int, default=10, help=argparse.SUPPRESS)
    parser.add_argument("--hidden-size", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--num-layers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--dropout", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=0.001, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=20, help=argparse.SUPPRESS)
    parser.add_argument("--num-runs", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--num-hpo-trials", type=int, default=10)
    parser.add_argument("--num-scoring-runs", type=int, default=10)
    parser.add_argument("--evaluations-per-trial", type=int, default=64)
    parser.add_argument("--output-dir", default="./data/output/rnn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    print(f"RNN test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(
        "RNN betting accuracy: "
        f"{result['betting_simulation_metrics']['accuracy']:.4f}"
    )
    print(f"Fixed betting final balance: {result['fixed_betting_final_balance']:.2f}")
    print(f"Kelly betting final balance: {result['kelly_betting_final_balance']:.2f}")


if __name__ == "__main__":
    main()
