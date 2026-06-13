"""Train PyOD models and generate XAI results for a saved dataset.

Expected input layout:

datasets/<dataset>/
    X_train.csv
    X_val.csv
    X_test.csv
    y_train.csv
    y_val.csv
    y_test.csv

Train and validation are combined for the final model fit. Test is used only
for evaluation and explanations.
"""

import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from .models import (
        usar_autoencoder,
        usar_cblof,
        usar_ecod,
        usar_hbos,
        usar_iforest,
        usar_mcd,
        usar_vae,
    )
except ImportError:
    from models import (
        usar_autoencoder,
        usar_cblof,
        usar_ecod,
        usar_hbos,
        usar_iforest,
        usar_mcd,
        usar_vae,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasets"
FALLBACK_DATASETS_DIR = PROJECT_ROOT / "prueba" / "data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "prueba" / "results" / "datasets"

MODEL_FUNCTIONS = {
    "CBLOF": usar_cblof,
    "IForest": usar_iforest,
    "ECOD": usar_ecod,
    "AutoEncoder": usar_autoencoder,
    "HBOS": usar_hbos,
    "MCD": usar_mcd,
    "VAE": usar_vae,
}

METRIC_COLUMNS = [
    "Modelo",
    "Semilla",
    "Normalizado",
    "Contaminacion",
    "TN",
    "FP",
    "FN",
    "TP",
    "Accuracy",
    "F1-score",
    "Sensibilidad",
    "Especificidad",
    "Precisión",
    "ROC-AUC",
    "Tiempo (s)",
]


def _find_csv(dataset_dir, stem):
    candidates = {
        path.stem.casefold(): path
        for path in dataset_dir.glob("*.csv")
    }
    aliases = [stem]
    if stem.startswith("y_"):
        aliases.extend([f"{stem}_phm", f"{stem}_phmd"])

    for alias in aliases:
        path = candidates.get(alias.casefold())
        if path is not None:
            return path

    expected = ", ".join(f"{alias}.csv" for alias in aliases)
    raise FileNotFoundError(
        f"No se encontró {expected} dentro de {dataset_dir}"
    )


def _load_target(path):
    target_df = pd.read_csv(path)
    if target_df.shape[1] != 1:
        unnamed = [
            column
            for column in target_df.columns
            if str(column).startswith("Unnamed:")
        ]
        target_df = target_df.drop(columns=unnamed)
    if target_df.shape[1] != 1:
        raise ValueError(
            f"{path} debe contener una única columna de etiquetas"
        )

    target = target_df.iloc[:, 0].astype(int).reset_index(drop=True)
    invalid_labels = set(target.unique()).difference({0, 1})
    if invalid_labels:
        raise ValueError(
            f"{path} contiene etiquetas distintas de 0 y 1: "
            f"{sorted(invalid_labels)}"
        )
    target.name = "target"
    return target


def load_dataset_splits(dataset_dir, drop_columns=None):
    dataset_dir = Path(dataset_dir)
    drop_columns = list(drop_columns or [])
    raw_splits = {}

    for split_name in ("train", "val", "test"):
        X_path = _find_csv(dataset_dir, f"X_{split_name}")
        y_path = _find_csv(dataset_dir, f"y_{split_name}")

        X = pd.read_csv(X_path)
        unnamed = [
            column for column in X.columns
            if str(column).startswith("Unnamed:")
        ]
        X = X.drop(
            columns=unnamed + drop_columns,
            errors="ignore",
        ).reset_index(drop=True)
        y = _load_target(y_path)

        if len(X) != len(y):
            raise ValueError(
                f"Dimensiones incompatibles en {split_name}: "
                f"X tiene {len(X)} filas e y tiene {len(y)}"
            )
        raw_splits[split_name] = (X, y)

    common_columns = set.intersection(
        *(set(X.columns) for X, _ in raw_splits.values())
    )
    target_leakage_columns = [
        column
        for column in common_columns
        if all(
            np.array_equal(
                np.asarray(X[column]).ravel(),
                y.to_numpy(),
            )
            for X, y in raw_splits.values()
        )
    ]
    if target_leakage_columns:
        print(
            "Se eliminan columnas idénticas al target en todos los splits: "
            f"{sorted(target_leakage_columns)}"
        )

    splits = {}
    for split_name, (X, y) in raw_splits.items():
        X = X.drop(columns=target_leakage_columns)
        non_numeric = list(
            X.select_dtypes(exclude=[np.number, "bool"]).columns
        )
        if non_numeric:
            print(
                f"X_{split_name}: se eliminan columnas no numéricas "
                f"{non_numeric}"
            )
            X = X.drop(columns=non_numeric)
        if X.isna().any().any():
            raise ValueError(f"X_{split_name} contiene valores NaN")

        splits[split_name] = (X.astype(float), y)

    train_columns = list(splits["train"][0].columns)
    for split_name in ("val", "test"):
        if list(splits[split_name][0].columns) != train_columns:
            raise ValueError(
                f"Las columnas de {split_name} no coinciden con train"
            )

    return splits


def resolve_dataset_dir(dataset, datasets_dir=None):
    if datasets_dir is not None:
        dataset_dir = Path(datasets_dir) / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"No existe la carpeta del dataset: {dataset_dir}"
            )
        return dataset_dir

    candidates = [
        DEFAULT_DATASETS_DIR / dataset,
        FALLBACK_DATASETS_DIR / dataset,
    ]
    for dataset_dir in candidates:
        if dataset_dir.is_dir():
            return dataset_dir

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No se encontró el dataset '{dataset}'. Rutas revisadas: {searched}"
    )


def prepare_training_data(splits, normalize=True):
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    X_fit = pd.concat([X_train, X_val], ignore_index=True)
    y_fit = pd.concat([y_train, y_val], ignore_index=True)
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    scaler = None
    if normalize:
        scaler = MinMaxScaler()
        X_fit = pd.DataFrame(
            scaler.fit_transform(X_fit),
            columns=X_fit.columns,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
        )

    contamination = float((y_fit == 1).mean())
    if not 0 < contamination <= 0.5:
        raise ValueError(
            "La contaminación de train+validación debe estar en (0, 0.5]. "
            f"Valor calculado: {contamination}"
        )

    return X_fit, y_fit, X_test, y_test, contamination, scaler


def select_observations(y_test, number, seed):
    if number <= 0:
        return []

    rng = np.random.default_rng(seed)
    anomaly_indices = y_test.index[y_test == 1].to_numpy()
    normal_indices = y_test.index[y_test == 0].to_numpy()

    anomaly_count = min(len(anomaly_indices), (number + 1) // 2)
    normal_count = min(len(normal_indices), number - anomaly_count)

    selected = []
    if anomaly_count:
        selected.extend(
            rng.choice(
                anomaly_indices,
                size=anomaly_count,
                replace=False,
            ).tolist()
        )
    if normal_count:
        selected.extend(
            rng.choice(
                normal_indices,
                size=normal_count,
                replace=False,
            ).tolist()
        )

    remaining = number - len(selected)
    if remaining:
        available = np.setdiff1d(
            y_test.index.to_numpy(),
            np.asarray(selected),
        )
        selected.extend(
            rng.choice(
                available,
                size=min(remaining, len(available)),
                replace=False,
            ).tolist()
        )

    return selected


def make_dataframe_safe(model):
    original_decision_function = model.decision_function

    def decision_function(X):
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        return original_decision_function(X)

    model.decision_function = decision_function
    return model


def save_attributions(
    attributions,
    observations,
    feature_names,
    output_path,
    y_test,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(
        np.asarray(attributions),
        index=observations,
        columns=feature_names,
    )
    result.index.name = "observation"
    result.insert(0, "target", y_test.loc[observations].to_numpy())
    result.to_csv(output_path)


def generate_explanations(
    model,
    X_background,
    X_test,
    y_test,
    observations,
    model_output_dir,
    methods,
):
    try:
        from .xai_methods.break_down import usar_breakdown
        from .xai_methods.lime import usar_lime
        from .xai_methods.shap import usar_shap_local
    except ImportError:
        from xai_methods.break_down import usar_breakdown
        from xai_methods.lime import usar_lime
        from xai_methods.shap import usar_shap_local

    method_functions = {
        "lime": usar_lime,
        "shap": usar_shap_local,
        "breakdown": usar_breakdown,
    }
    failures = []

    for method_name in methods:
        print(f"  XAI: {method_name}")
        try:
            attributions = method_functions[method_name](
                model,
                X_background,
                X_test,
                observations,
            )
            save_attributions(
                attributions=attributions,
                observations=observations,
                feature_names=X_test.columns,
                output_path=(
                    model_output_dir
                    / "attributions"
                    / f"{method_name}.csv"
                ),
                y_test=y_test,
            )
        except Exception as error:
            failures.append(
                {
                    "stage": f"xai:{method_name}",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"  Error en {method_name}: {error}")

    return failures


def run_experiments(args):
    dataset_dir = resolve_dataset_dir(
        args.dataset,
        datasets_dir=args.datasets_dir,
    )
    output_dir = Path(args.results_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_dataset_splits(
        dataset_dir,
        drop_columns=args.drop_columns,
    )
    (
        X_fit,
        y_fit,
        X_test,
        y_test,
        contamination,
        scaler,
    ) = prepare_training_data(
        splits,
        normalize=not args.no_normalize,
    )

    observations = select_observations(
        y_test,
        number=args.xai_observations,
        seed=args.seed,
    )
    background = X_fit
    if len(background) > args.background_size:
        background = background.sample(
            n=args.background_size,
            random_state=args.seed,
        )

    run_metadata = {
        "dataset": args.dataset,
        "dataset_dir": str(dataset_dir.resolve()),
        "models": args.models,
        "xai_methods": args.xai_methods,
        "seed": args.seed,
        "normalized": not args.no_normalize,
        "contamination_train_val": contamination,
        "train_rows": len(splits["train"][0]),
        "validation_rows": len(splits["val"][0]),
        "fit_rows": len(X_fit),
        "test_rows": len(X_test),
        "features": list(X_fit.columns),
        "drop_columns": args.drop_columns,
        "xai_observations": observations,
        "background_size": len(background),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2, ensure_ascii=False)

    if scaler is not None:
        with (output_dir / "scaler.pkl").open("wb") as file:
            cloudpickle.dump(scaler, file)

    metrics = pd.DataFrame(columns=METRIC_COLUMNS)
    failures = []

    for model_name in args.models:
        print(f"\n********** {model_name} **********")
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            metrics, model = MODEL_FUNCTIONS[model_name](
                X_fit,
                y_fit,
                X_test,
                y_test,
                metrics,
                not args.no_normalize,
                contamination,
                args.seed,
            )
            model = make_dataframe_safe(model)

            with (model_output_dir / "model.pkl").open("wb") as file:
                cloudpickle.dump(model, file)

            predictions = pd.DataFrame(
                {
                    "target": y_test,
                    "prediction": model.predict(X_test),
                    "anomaly_score": model.decision_function(X_test),
                }
            )
            predictions.to_csv(
                model_output_dir / "test_predictions.csv",
                index_label="observation",
            )

            failures.extend(
                {
                    "model": model_name,
                    **failure,
                }
                for failure in generate_explanations(
                    model=model,
                    X_background=background,
                    X_test=X_test,
                    y_test=y_test,
                    observations=observations,
                    model_output_dir=model_output_dir,
                    methods=args.xai_methods,
                )
            )
        except Exception as error:
            failures.append(
                {
                    "model": model_name,
                    "stage": "training",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(f"Error entrenando {model_name}: {error}")
        finally:
            metrics.to_csv(output_dir / "test_metrics.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(
            output_dir / "failures.csv",
            index=False,
        )

    print(f"\nResultados guardados en: {output_dir}")
    return metrics, failures


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Entrena modelos PyOD con train+validación, evalúa test y "
            "genera explicaciones LIME, SHAP y BreakDown."
        )
    )
    parser.add_argument(
        "dataset",
        help="Nombre de la carpeta situada dentro de datasets/",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=None,
        help=(
            "Directorio raíz de datasets. Si se omite, busca en datasets/ "
            "y después en prueba/data/."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_FUNCTIONS),
        default=list(MODEL_FUNCTIONS),
    )
    parser.add_argument(
        "--xai-methods",
        nargs="+",
        choices=["lime", "shap", "breakdown"],
        default=["lime", "shap", "breakdown"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--xai-observations",
        type=int,
        default=10,
        help="Número de observaciones de test que explica cada método.",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=200,
        help="Máximo de filas de train+validación usadas como fondo XAI.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Usa los CSV tal como están, sin aplicar MinMaxScaler.",
    )
    parser.add_argument(
        "--drop-columns",
        nargs="*",
        default=[],
        help=(
            "Columnas identificadoras que deben excluirse, especialmente "
            "si son numéricas."
        ),
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
