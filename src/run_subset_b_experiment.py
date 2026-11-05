"""
Experiment: test paper's main-text Subset B calibration features.

The paper (Section 8) lists Subset B as: 3P, BLK, FT%, ORB, AST%, Previous Season Win Pct.
Our pipeline's feature selection produced {ORtg, DRB, AST} (matches supplemental Table 4).

This script bypasses feature selection and HPO, using:
  - Calibration features: paper's Subset B (substituting ORB% for ORB, which was dropped
    by covariate shift detection in our pipeline)
  - HPO hyperparameters: loaded from cached JSON (accuracy branch unchanged)

Outputs are saved to data/output/subset_b_experiment/ to avoid overwriting main results.
"""
import os
import sys
import json
import time
import logging
import numpy as np

# add src to path so imports work when run from project root
sys.path.insert(0, os.path.join('.', 'src'))

from pipeline import NBABettingExperiment
from model_selection import ModelSelector
from simulate import BettingSimulation

logging.basicConfig(level=logging.INFO, format='%(message)s')

OUTPUT_DIR = os.path.join('.', 'data', 'output', 'subset_b_experiment')

# Paper's Subset B — ORB replaced with ORB% (ORB dropped by KS covariate shift test)
SUBSET_B_FEATURES = ['3P', 'BLK', 'FT%', 'ORB%', 'AST%', 'previous_win_pct']


def load_cached_hyperparameters():
    with open(os.path.join('.', 'data', 'output', 'hpo', 'calibration_driven_optimal_hyperparameters.json')) as f:
        calibration_hps = json.load(f)
    with open(os.path.join('.', 'data', 'output', 'hpo', 'accuracy_driven_optimal_hyperparameters.json')) as f:
        accuracy_hps = json.load(f)
    return accuracy_hps, calibration_hps


def run_model_selection(preprocessed_datasets, modelling_ready_datasets, features, hyperparameters, metric, num_runs):
    extended_X = modelling_ready_datasets.extended_training
    test_X = modelling_ready_datasets.test
    extended_y = preprocessed_datasets.extended_training['home_victory'].to_list()
    test_y = preprocessed_datasets.test['home_victory'].to_list()

    selector = ModelSelector(
        extended_X, test_X, extended_y, test_y,
        features, hyperparameters,
        metric=metric,
        num_scoring_runs=num_runs,
    )
    selector.score_lr()
    selector.score_rf()
    selector.score_svm()
    selector.score_mlp()

    logging.info(f'\n{metric}-driven model scores (Subset B features):')
    for model, score in selector.model_scores.items():
        logging.info(f'  {model}: {round(100 * score, 4)}%')

    best_model = selector.extract_best_model()
    logging.info(f'Best {metric}-driven model: {best_model}')

    with open(os.path.join(OUTPUT_DIR, f'{metric}_driven_model_scores.json'), 'w') as f:
        json.dump(selector.model_scores, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, f'{metric}_driven_best_model.json'), 'w') as f:
        json.dump({'best_model': best_model}, f, indent=4)

    return best_model


def run_simulation(preprocessed_datasets, modelling_ready_datasets, features, hyperparameters, model, metric, num_runs):
    final_X = modelling_ready_datasets.final_training
    betting_X = modelling_ready_datasets.betting_simulation
    final_y = preprocessed_datasets.final_training['home_victory'].to_list()
    betting_y = preprocessed_datasets.betting_simulation['home_victory'].to_list()
    odds = preprocessed_datasets.odds

    sim = BettingSimulation(
        model, features, hyperparameters,
        final_X, final_y, betting_X, betting_y, odds,
        metric=metric, num_runs=num_runs,
    )
    sim.generate_predictions()

    for rule in ['fixed', 'kelly']:
        bankroll_tracker, _ = sim.run_betting_simulation(rule=rule, save_results=False)
        final_balance = bankroll_tracker[-1]
        roi = round(100 * (final_balance - 10_000) / 10_000, 2)
        logging.info(f'{metric}-driven {rule} ROI: {roi}%')
        with open(os.path.join(OUTPUT_DIR, f'{metric}_driven_{rule}_roi.json'), 'w') as f:
            json.dump({'ROI': roi, 'final_balance': final_balance}, f, indent=4)


def main():
    NUM_PREDICTION_RUNS = 10

    experiment = NBABettingExperiment(num_prediction_runs=NUM_PREDICTION_RUNS)

    logging.info('\nStep 1: Loading and preprocessing data...')
    preprocessed_datasets = experiment.load_and_preprocess()

    logging.info('\nStep 2: Extracting features...')
    modelling_ready_datasets = experiment.extract_features(preprocessed_datasets)

    logging.info('\nStep 3: Loading cached HPO hyperparameters (skipping HPO)...')
    accuracy_hps, calibration_hps = load_cached_hyperparameters()
    logging.info(f'Calibration HPO loaded: {list(calibration_hps.keys())}')

    logging.info(f'\nStep 4: Running calibration model selection with Subset B features: {SUBSET_B_FEATURES}')
    cal_best = run_model_selection(
        preprocessed_datasets, modelling_ready_datasets,
        SUBSET_B_FEATURES, calibration_hps, 'calibration', NUM_PREDICTION_RUNS
    )

    logging.info(f'\nStep 5: Running betting simulation with {cal_best} (calibration, Subset B)...')
    run_simulation(
        preprocessed_datasets, modelling_ready_datasets,
        SUBSET_B_FEATURES, calibration_hps, cal_best, 'calibration', NUM_PREDICTION_RUNS
    )

    logging.info('\nDone. Results saved to data/output/subset_b_experiment/')


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'\nExperiment took {round((time.time() - start) / 60, 1)} minutes.')
