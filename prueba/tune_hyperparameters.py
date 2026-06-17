"""Tune PyOD models and their decision threshold for validation F1."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.mcd import MCD
from pyod.models.vae import VAE

try:
    from .run_dataset_experiments import (
        DEFAULT_RESULTS_DIR,
        load_dataset_splits,
        resolve_dataset_dir,
    )
except ImportError:
    from run_dataset_experiments import (
        DEFAULT_RESULTS_DIR,
        load_dataset_splits,
        resolve_dataset_dir,
    )


MODEL_CLASSES = {
    "CBLOF": CBLOF,
    "IForest": IForest,
    "ECOD": ECOD,
    "AutoEncoder": AutoEncoder,
    "HBOS": HBOS,
    "MCD": MCD,
    "VAE": VAE,
}

PARAMETER_GRIDS = {
    "CBLOF": {
        "n_clusters": [4, 8, 12],
        "alpha": [0.8, 0.9],
        "beta": [3, 5],
        "use_weights": [False, True],
    },
    "IForest": {
        "n_estimators": [100, 300],
        "max_samples": [256, 1024, "auto"],
        "max_features": [0.7, 1.0],
        "bootstrap": [False, True],
    },
    "ECOD": {},
    "AutoEncoder": {
        "hidden_neuron_list": [[32, 16], [64, 32], [128, 64, 32]],
        "lr": [0.0005, 0.001],
        "epoch_num": [10, 20],
        "batch_size": [64, 256],
        "dropout_rate": [0.1, 0.2],
    },
    "HBOS": {
        "n_bins": [10, 20, 50, "auto"],
        "alpha": [0.05, 0.1],
        "tol": [0.3, 0.5],
    },
    "MCD": {
        "support_fraction": [None, 0.7, 0.9],
        "assume_centered": [False, True],
    },
    "VAE": {
        "encoder_neuron_list": [[64, 32], [128, 64, 32]],
        "decoder_neuron_list": [[32, 64], [32, 64, 128]],
        "latent_dim": [2, 4],
        "beta": [0.5, 1.0],
        "lr": [0.0005, 0.001],
        "epoch_num": [15, 30],
        "batch_size": [64, 256],
    },
}

NORMAL_ONLY_MODELS = {"AutoEncoder", "VAE"}


def prepare_tuning_data(splits, normalize=True):
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler = None
    if normalize:
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
        )
        X_val = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
        )

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def best_f1_threshold(y_true, anomaly_scores):
    y_true = np.asarray(y_true, dtype=int).ravel()
    anomaly_scores = np.asarray(anomaly_scores, dtype=float).ravel()

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        anomaly_scores,
    )
    if thresholds.size == 0:
        return float("inf"), 0.0, 0.0, 0.0

    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1_values))

    return (
        float(thresholds[best_index]),
        float(f1_values[best_index]),
        float(precision[best_index]),
        float(recall[best_index]),
    )


def _fit_model(model_name, params, X, y, contamination, seed):
    constructor_params = dict(params)
    constructor_params["contamination"] = contamination

    if model_name in {"CBLOF", "IForest", "AutoEncoder", "MCD", "VAE"}:
        constructor_params["random_state"] = seed
    if model_name in {"AutoEncoder", "VAE"}:
        constructor_params["verbose"] = 0
    if model_name == "ECOD":
        constructor_params["n_jobs"] = 1

    model = MODEL_CLASSES[model_name](**constructor_params)
    X_model = X[y == 0] if model_name in NORMAL_ONLY_MODELS else X
    model.fit(np.asarray(X_model))
    return model


def _candidate_parameters(model_name, max_trials, seed):
    candidates = list(ParameterGrid(PARAMETER_GRIDS[model_name]))
    if max_trials is None or len(candidates) <= max_trials:
        return candidates

    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(candidates),
        size=max_trials,
        replace=False,
    )
    return [candidates[index] for index in sorted(indices)]


def tune_model(
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    seed=42,
    max_trials=20,
):
    base_contamination = float((y_train == 1).mean())
    candidates = _candidate_parameters(model_name, max_trials, seed)
    trial_rows = []

    for trial_number, params in enumerate(candidates, start=1):
        started = perf_counter()
        print(
            f"  Trial {trial_number}/{len(candidates)}: "
            f"{json.dumps(params, sort_keys=True)}"
        )
        try:
            model = _fit_model(
                model_name,
                params,
                X_train,
                y_train,
                contamination=base_contamination,
                seed=seed,
            )
            train_scores = model.decision_function(np.asarray(X_train))
            val_scores = model.decision_function(np.asarray(X_val))
            threshold, val_f1, val_precision, val_recall = (
                best_f1_threshold(y_val, val_scores)
            )

            tuned_contamination = float(np.mean(train_scores >= threshold))
            minimum_contamination = 1.0 / len(train_scores)
            tuned_contamination = float(
                np.clip(tuned_contamination, minimum_contamination, 0.5)
            )

            trial_rows.append(
                {
                    "trial": trial_number,
                    "status": "ok",
                    "params": json.dumps(params, sort_keys=True),
                    "validation_f1": val_f1,
                    "validation_precision": val_precision,
                    "validation_recall": val_recall,
                    "validation_threshold": threshold,
                    "tuned_contamination": tuned_contamination,
                    "duration_seconds": perf_counter() - started,
                    "error": "",
                }
            )
        except Exception as error:
            trial_rows.append(
                {
                    "trial": trial_number,
                    "status": "error",
                    "params": json.dumps(params, sort_keys=True),
                    "validation_f1": np.nan,
                    "validation_precision": np.nan,
                    "validation_recall": np.nan,
                    "validation_threshold": np.nan,
                    "tuned_contamination": np.nan,
                    "duration_seconds": perf_counter() - started,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    trials = pd.DataFrame(trial_rows)
    successful = trials[trials["status"] == "ok"]
    if successful.empty:
        raise RuntimeError(
            f"Ninguna combinación funcionó para {model_name}"
        )

    best_index = successful["validation_f1"].idxmax()
    best_trial = trials.loc[best_index]
    return trials, best_trial


def _binary_metrics(y_true, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Accuracy": accuracy_score(y_true, predictions),
        "F1-score": f1_score(y_true, predictions, zero_division=0),
        "Sensibilidad": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Precisión": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC-score": roc_auc_score(y_true, scores),
    }


def fit_selected_and_evaluate(
    model_name,
    best_trial,
    splits,
    normalize=True,
    seed=42,
):
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler = None
    if normalize:
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
        )
        X_val = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
        )

    params = json.loads(best_trial["params"])
    base_contamination = float((y_train == 1).mean())
    started = perf_counter()
    model = _fit_model(
        model_name,
        params,
        X_train,
        y_train,
        contamination=base_contamination,
        seed=seed,
    )
    train_scores = model.decision_function(np.asarray(X_train))
    validation_scores = model.decision_function(np.asarray(X_val))
    (
        calibrated_threshold,
        validation_f1,
        validation_precision,
        validation_recall,
    ) = best_f1_threshold(y_val, validation_scores)
    model.threshold_ = calibrated_threshold

    effective_contamination = float(
        np.mean(train_scores >= calibrated_threshold)
    )
    predictions = model.predict(np.asarray(X_test))
    scores = model.decision_function(np.asarray(X_test))

    metrics = {
        "Modelo": model_name,
        "validation_f1": validation_f1,
        "validation_precision": validation_precision,
        "validation_recall": validation_recall,
        "calibrated_threshold": calibrated_threshold,
        "effective_contamination": effective_contamination,
        "best_params": best_trial["params"],
        "duration_seconds": perf_counter() - started,
        **_binary_metrics(y_test, predictions, scores),
    }
    prediction_frame = pd.DataFrame(
        {
            "target": y_test,
            "prediction": predictions,
            "anomaly_score": scores,
        }
    )
    return model, scaler, metrics, prediction_frame


def run_tuning(args):
    dataset_dir = resolve_dataset_dir(
        args.dataset,
        datasets_dir=args.datasets_dir,
    )
    splits = load_dataset_splits(
        dataset_dir,
        drop_columns=args.drop_columns,
    )
    (
        X_train,
        y_train,
        X_val,
        y_val,
        _,
        _,
        _,
    ) = prepare_tuning_data(
        splits,
        normalize=not args.no_normalize,
    )

    output_root = (
        Path(args.results_dir)
        / args.dataset
        / "tuning"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "tuned_test_metrics.csv"
    if metrics_path.exists():
        previous_metrics = pd.read_csv(metrics_path)
        previous_metrics = previous_metrics[
            ~previous_metrics["Modelo"].isin(args.models)
        ]
        test_metrics = previous_metrics.to_dict("records")
    else:
        test_metrics = []

    for model_name in args.models:
        print(f"\n********** Tuning {model_name} **********")
        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        trials, best_trial = tune_model(
            model_name,
            X_train,
            y_train,
            X_val,
            y_val,
            seed=args.seed,
            max_trials=args.max_trials,
        )
        trials.to_csv(model_dir / "validation_trials.csv", index=False)

        model, scaler, metrics, predictions = fit_selected_and_evaluate(
            model_name,
            best_trial,
            splits,
            normalize=not args.no_normalize,
            seed=args.seed,
        )
        with (model_dir / "tuned_model.pkl").open("wb") as file:
            cloudpickle.dump(model, file)
        if scaler is not None:
            with (model_dir / "scaler.pkl").open("wb") as file:
                cloudpickle.dump(scaler, file)

        predictions.to_csv(
            model_dir / "test_predictions.csv",
            index_label="observation",
        )
        with (model_dir / "best_parameters.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "params": json.loads(best_trial["params"]),
                    "calibrated_threshold": float(
                        metrics["calibrated_threshold"]
                    ),
                    "effective_contamination": float(
                        metrics["effective_contamination"]
                    ),
                    "validation_f1": metrics["validation_f1"],
                    "validation_precision": metrics[
                        "validation_precision"
                    ],
                    "validation_recall": metrics["validation_recall"],
                },
                file,
                indent=2,
            )

        test_metrics.append(metrics)
        pd.DataFrame(test_metrics).to_csv(metrics_path, index=False)

    print(f"\nResultados guardados en: {output_root}")
    return pd.DataFrame(test_metrics)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Optimiza hiperparámetros y calibra el umbral mediante F1 de "
            "validación; evalúa test solo tras congelar el ganador."
        )
    )
    parser.add_argument("dataset")
    parser.add_argument("--datasets-dir", type=Path, default=None)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CLASSES),
        default=["IForest", "ECOD", "HBOS", "CBLOF"],
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=20,
        help="Máximo de combinaciones por modelo.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--drop-columns", nargs="*", default=[])
    return parser.parse_args(argv)


def main():
    run_tuning(parse_args())


if __name__ == "__main__":
    main()
