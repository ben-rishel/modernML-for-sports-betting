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
    parser.add_argument("--num-hpo-trials", type=int, default=10)
    parser.add_argument("--num-scoring-runs", type=int, default=10)
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

    print(
        "CatBoost 2018 test calibration score: "
        f"{result['test_metrics']['calibration_score']:.6f}"
    )
    print(
        "CatBoost 2018 test raw class-wise ECE: "
        f"{result['test_metrics']['raw_calibration_score']:.6f}"
    )
    print(
        "2018 runs triggering the sparse-bin fallback: "
        f"{result['test_metrics']['calibration_guard_failures']} / "
        f"{len(result['test_metrics']['populated_bins_per_run'])}"
    )
    print(f"CatBoost 2018 test accuracy: {result['test_metrics']['accuracy']:.4f}")
    print(f"2019 fixed betting ROI: {result['fixed_betting_roi']:.2f}%")
    print(f"2019 Kelly betting ROI: {result['kelly_betting_roi']:.2f}%")
    print(f"Selected hyperparameters: {result['hyperparameters']}")


if __name__ == "__main__":
    main()
