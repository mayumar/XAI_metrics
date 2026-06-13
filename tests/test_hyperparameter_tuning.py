import numpy as np
import pandas as pd


def test_best_f1_threshold_finds_perfect_separation():
    from prueba.tune_hyperparameters import best_f1_threshold

    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    threshold, f1, precision, recall = best_f1_threshold(
        y_true,
        scores,
    )

    assert threshold == 0.8
    assert f1 == 1.0
    assert precision == 1.0
    assert recall == 1.0


def test_tune_model_uses_validation_f1():
    from prueba.tune_hyperparameters import tune_model

    X_train = pd.DataFrame(
        {
            "a": [0.0, 0.1, 0.2, 0.3, 2.0],
            "b": [0.0, 0.1, 0.0, 0.2, 2.0],
        }
    )
    y_train = pd.Series([0, 0, 0, 0, 1])
    X_val = pd.DataFrame(
        {
            "a": [0.05, 0.15, 1.8, 2.2],
            "b": [0.05, 0.15, 1.9, 2.1],
        }
    )
    y_val = pd.Series([0, 0, 1, 1])

    trials, best_trial = tune_model(
        "IForest",
        X_train,
        y_train,
        X_val,
        y_val,
        seed=42,
        max_trials=1,
    )

    assert len(trials) == 1
    assert best_trial["status"] == "ok"
    assert 0.0 <= best_trial["validation_f1"] <= 1.0
    assert 0.0 < best_trial["tuned_contamination"] <= 0.5
