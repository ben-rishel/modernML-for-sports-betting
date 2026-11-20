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

2. **Install Python 3.12** (Python 3.10–3.12 is supported; Python 3.13 is intentionally excluded because of dependency compatibility issues):

   **Windows:**
   ```powershell
   winget install --id Python.Python.3.12 --exact --source winget
   py -3.12 --version
   ```

   **macOS/Linux:** install Python 3.12 with your system package manager or a version manager such as pyenv, then verify:
   ```bash
   python3.12 --version
   ```

3. **Install and configure Poetry** (requires [Poetry](https://python-poetry.org/)):
   ```bash
   python -m pip install --user --upgrade pipx
   python -m pipx ensurepath
   ```

   Close and reopen your editor/terminal, then run:

   ```bash
   pipx install poetry==2.4.2
   poetry --version
   ```

   Select the Python 3.12 interpreter before installing dependencies:

   **Windows:**
   ```powershell
   poetry env use (py -3.12 -c "import sys; print(sys.executable)")
   ```

   **macOS/Linux:**
   ```bash
   poetry env use "$(command -v python3.12)"
   ```

   **Install Dependencies**
   ```bash
   poetry install
   ```

   If you previously created a broken environment, remove it and reinstall:

   ```bash
   poetry env remove --all
   poetry install
   ```

   On Windows, if virtualenv reports `WinError 5: Access is denied`, close other terminals and VS Code instances, clear its cache, and recreate the environment:

   ```powershell
   Remove-Item -Recurse -Force "$env:LOCALAPPDATA\pypa\virtualenv\Cache" -ErrorAction SilentlyContinue
   poetry env remove --all
   poetry env use (py -3.12 -c "import sys; print(sys.executable)")
   poetry install
   ```

4. **Run a pipeline** — see commands above.


---

## Bankroll Chart

To regenerate the bankroll progression chart (`data/output/bankroll_chart.png`) showing all four models across the 2018-19 betting simulation:

```bash
poetry run python src/run_all_and_plot.py
```

This script re-runs the SVM betting simulation using saved HPO results (~2 minutes), re-runs the RNN pipeline (~2–3 hours), then generates the chart automatically. CatBoost bankroll data is read from the existing results and does not require a re-run.

The chart is already committed at `data/output/bankroll_chart.png` if you only need to view it.

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
