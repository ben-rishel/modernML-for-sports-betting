"""Run SVM betting simulation (using saved HPO results), RNN pipeline, then generate bankroll chart."""

import json
import os
import time


def run_svm_simulation_only():
    """Re-run just the betting simulation using saved HPO/model-selection results (~2 min)."""
    print("=" * 60)
    print("STEP 1/3: Running SVM betting simulation (using saved HPO results)")
    print("=" * 60)

    from pipeline import NBABettingExperiment
    from run_pipeline_subset_b import SUBSET_B_FEATURES, SUBSET_C_FEATURES

    trial = NBABettingExperiment(
        num_hpo_trials=10,
        num_prediction_runs=10,
        calibration_feature_override=SUBSET_B_FEATURES,
        accuracy_feature_override=SUBSET_C_FEATURES,
    )
    preprocessed = trial.load_and_preprocess()
    modelling_ready = trial.extract_features(preprocessed)

    # Load saved HPO results — skip HPO and model selection entirely
    from collections import namedtuple

    with open("data/output/hpo/calibration_driven_optimal_hyperparameters.json") as f:
        cal_hpo = json.load(f)
    with open("data/output/hpo/accuracy_driven_optimal_hyperparameters.json") as f:
        acc_hpo = json.load(f)
    OptimalHyperparameters = namedtuple("OptimalHyperparameters", ["accuracy_driven", "calibration_driven"])
    optimal_hyperparameters = OptimalHyperparameters(accuracy_driven=acc_hpo, calibration_driven=cal_hpo)

    OptimalFeatures = namedtuple("OptimalFeatures", ["accuracy_driven", "calibration_driven"])
    optimal_features = OptimalFeatures(accuracy_driven=SUBSET_C_FEATURES, calibration_driven=SUBSET_B_FEATURES)

    OptimalModels = namedtuple("OptimalModels", ["accuracy_driven", "calibration_driven"])
    optimal_models = OptimalModels(accuracy_driven="svm", calibration_driven="svm")

    betting_simulation_plotting_data = trial.run_betting_simulations(
        preprocessed, modelling_ready, optimal_features, optimal_hyperparameters, optimal_models
    )
    trial.analyse_results(betting_simulation_plotting_data)
    print("SVM simulation done.")


def run_rnn_pipeline():
    print("=" * 60)
    print("STEP 2/3: Running RNN pipeline (~2-3 hours)")
    print("=" * 60)
    from run_rnn import run_experiment

    class Args:
        initial_training_years = [2015, 2016]
        validation_years = [2017]
        test_years = [2018]
        betting_simulation_years = [2019]
        num_hpo_trials = 10
        num_scoring_runs = 5
        evaluations_per_trial = 20
        output_dir = "./data/output/rnn"

    run_experiment(Args())


def generate_chart():
    print("=" * 60)
    print("STEP 3/3: Generating bankroll chart")
    print("=" * 60)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    with open("data/output/fixed_betting/calibration_driven_bankroll.json") as f:
        svm_cal_fixed = json.load(f)
    with open("data/output/kelly_betting/calibration_driven_bankroll.json") as f:
        svm_cal_kelly = json.load(f)
    with open("data/output/fixed_betting/accuracy_driven_bankroll.json") as f:
        svm_acc_fixed = json.load(f)
    with open("data/output/kelly_betting/accuracy_driven_bankroll.json") as f:
        svm_acc_kelly = json.load(f)
    with open("data/output/catboost/catboost_betting_simulation_data.json") as f:
        cb_data = json.load(f)
    with open("data/output/rnn/fixed_bankroll.json") as f:
        rnn_fixed = json.load(f)
    with open("data/output/rnn/kelly_bankroll.json") as f:
        rnn_kelly = json.load(f)

    colors = {
        "Cal. SVM":  "#1f77b4",
        "Acc. SVM":  "#d62728",
        "CatBoost":  "#2ca02c",
        "RNN":       "#ff7f0e",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    fixed_series = [
        ("Cal. SVM",  svm_cal_fixed),
        ("Acc. SVM",  svm_acc_fixed),
        ("CatBoost",  cb_data["fixed_betting_bankroll"]),
        ("RNN",       rnn_fixed),
    ]
    kelly_series = [
        ("Cal. SVM",  svm_cal_kelly),
        ("Acc. SVM",  svm_acc_kelly),
        ("CatBoost",  cb_data["kelly_betting_bankroll"]),
        ("RNN",       rnn_kelly),
    ]

    for label, bankroll in fixed_series:
        ax1.plot(bankroll, label=label, color=colors[label], linewidth=1.5)
    for label, bankroll in kelly_series:
        ax2.plot(bankroll, label=label, color=colors[label], linewidth=1.5)

    for ax, title in [(ax1, "Fixed Betting ($100/bet)"), (ax2, "⅛-Kelly Betting")]:
        ax.axhline(10000, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Game", fontsize=11)
        ax.set_ylabel("Bankroll ($)", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Bankroll Progression — 2018-19 Betting Simulation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join("data", "output", "bankroll_chart.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    start = time.time()
    run_svm_simulation_only()
    run_rnn_pipeline()
    generate_chart()
    elapsed = (time.time() - start) / 60
    print(f"\nAll done in {elapsed:.1f} minutes.")
    print("Chart saved to data/output/bankroll_chart.png")
