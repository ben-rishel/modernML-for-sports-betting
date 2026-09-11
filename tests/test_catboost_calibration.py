from catboost_calibration import (
    CatBoostCalibrationHPO,
    calculate_calibration_score,
)


def test_calibration_score_is_finite():
    y_true = [0, 0, 1, 1]
    y_score = [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.3, 0.7],
    ]

    score = calculate_calibration_score(y_true, y_score)

    assert isinstance(score, float)
    assert score >= 0


def test_catboost_hpo_returns_hyperparameters():
    X_train = [[0.1, 0.1], [0.2, 0.2], [0.9, 0.9], [1.0, 1.0]]
    X_validate = [[0.15, 0.15], [0.85, 0.85]]
    y_train = [0, 0, 1, 1]
    y_validate = [0, 1]

    hpo = CatBoostCalibrationHPO(
        X_train,
        X_validate,
        y_train,
        y_validate,
        num_trials=2,
        num_scoring_runs=1,
    )

    params = hpo.run_hpo()

    assert isinstance(params, dict)
    assert {"depth", "learning_rate", "iterations"}.issubset(set(params.keys()))
