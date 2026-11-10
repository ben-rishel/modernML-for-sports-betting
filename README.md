# Modern ML for Sports Betting: Extending Walsh & Joshi (2024)

## Overview

This repository is a DS340W project extending ["Machine learning for sports betting: Should model selection be based on accuracy or calibration?"](https://doi.org/10.1016/j.mlwa.2024.100539) (Walsh & Joshi, 2024).

The parent paper demonstrates that calibration-driven model selection, choosing models and features based on probability calibration (ECE) rather than classification accuracy, produces significantly better betting returns using an SVM baseline. This project asks whether modern ML architectures (CatBoost and RNN) improve on those results when applied within the same calibration-driven framework.

**Team:**
- Ben Rishel — CatBoost pipeline
- Jacqueline Maloney — RNN pipeline

---

## Pipelines

There are three runnable pipelines:

### 1. Parent paper reproduction (`run_pipeline_subset_b.py`)
Reproduces the Walsh & Joshi (2024) results using the paper's main-text feature subsets (Subset B for calibration, Subset C for accuracy). Runs SVM, LR, RF, and MLP candidates and selects the best under each criterion.

```bash
poetry run python src/run_pipeline_subset_b.py
```

Runtime: ~45 minutes. Results saved to `data/output/`.

### 2. CatBoost calibration benchmark (`run_catboost_calibration.py`)
Runs a standalone CatBoost experiment using the same calibration-driven protocol as the parent paper (Subset B features, ECE-based HPO, identical data splits).

```bash
poetry run python src/run_catboost_calibration.py
```

Runtime: ~15–20 minutes. Results saved to `data/output/catboost/`.

### 3. RNN calibration benchmark (`run_rnn.py`)
Runs a standalone GRU-based RNN experiment. Uses raw per-team box score time series (all available features) with team-aware sequential data construction, ECE-based HPO, and identical data splits.

```bash
poetry run python src/run_rnn.py
```

Runtime: ~2–3 hours. Results saved to `data/output/rnn/`.

Key options:
```bash
poetry run python src/run_rnn.py --num-hpo-trials 10 --evaluations-per-trial 20 --num-scoring-runs 5
```

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd modernML-for-sports-betting
   ```

2. **Install dependencies** (requires [Poetry](https://python-poetry.org/)):
   ```bash
   poetry install
   ```

3. **Run a pipeline** — see commands above.

---

## Key Results

| Metric | SVM (reproduced) | CatBoost | RNN (GRU) |
|--------|-----------------|----------|-----------|
| Test ECE | 3.17% | 2.65% | 5.52% |
| Test Accuracy | 66.6% | 63.6% | 63.0% |
| Fixed Betting ROI (2019) | +45.1% | +43.6% | +15.1% |
| Kelly Betting ROI (2019) | +52.3% | +96.7% | −10.1% |

Paper reported values: Calibration Kelly +36.93%, Accuracy Kelly −75.9%.

---

## Data Splits

All pipelines use the same temporal splits to prevent data leakage:

| Split | Years | Purpose |
|-------|-------|---------|
| Initial training | 2015–2016 | Model training during HPO |
| Validation | 2017 | HPO scoring (ECE or accuracy) |
| Test | 2018 | Final model evaluation |
| Betting simulation | 2019 | ROI measurement |

---

## Parent Paper

Walsh, C., & Joshi, A. (2024). Machine learning for sports betting: should model selection be based on accuracy or calibration? *Machine Learning with Applications*, 16, 100539. [https://doi.org/10.1016/j.mlwa.2024.100539](https://doi.org/10.1016/j.mlwa.2024.100539)
