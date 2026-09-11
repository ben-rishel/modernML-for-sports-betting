import argparse

from pipeline import NBABettingExperiment
from catboost_calibration import run_catboost_calibration_side_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run a standalone CatBoost calibration benchmark using the repo's existing data splits."
    )
    parser.add_argument("--initial-training-years", nargs="*", type=int, default=[2015, 2016])
    parser.add_argument("--validation-years", nargs="*", type=int, default=[2017])
    parser.add_argument("--test-years", nargs="*", type=int, default=[2018])
    parser.add_argument("--betting-simulation-years", nargs="*", type=int, default=[2019])
    parser.add_argument("--num-hpo-trials", type=int, default=4)
    parser.add_argument("--num-scoring-runs", type=int, default=2)
    parser.add_argument("--output-dir", default="./data/output/catboost")
    args = parser.parse_args()

    experiment = NBABettingExperiment(
        initial_training_years=args.initial_training_years,
        validation_years=args.validation_years,
        test_years=args.test_years,
        betting_simulation_years=args.betting_simulation_years,
        num_hpo_trials=args.num_hpo_trials,
        num_prediction_runs=args.num_scoring_runs,
    )

    result = run_catboost_calibration_side_experiment(
        experiment,
        metric="calibration",
        output_dir=args.output_dir,
    )

    print(f"CatBoost calibration score: {result['calibration_score']:.6f}")
    print(f"Selected hyperparameters: {result['hyperparameters']}")


if __name__ == "__main__":
    main()
