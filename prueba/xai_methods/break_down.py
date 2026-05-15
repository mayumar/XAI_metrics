import pandas as pd
import numpy as np
import dalex as dx
import torch
from pathlib import Path

from utils import QuantusWrapper
from config import DATASETS

from XAI_metrics.base import MetricContext
from XAI_metrics.runner import run_all_metrics
from XAI_metrics.reporting import save_metrics_report

from pyod.models.base import BaseDetector
from typing import List

def _predict_anomaly_score(model, X):
    if isinstance(X, pd.DataFrame):
        X_input = X
    else:
        X_input = pd.DataFrame(X)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_input), dtype=float).ravel()
    if hasattr(model, "model") and hasattr(model.model, "decision_function"):
        return np.asarray(model.model.decision_function(X_input), dtype=float).ravel()
    
    raise AttributeError(
        f"No se encontro decision_function en {type(model)} ni en model.model"
    )


def _breakdown_attributions(
    explainer: dx.Explainer,
    X: pd.DataFrame,
    cols: List[str]
):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=cols)

    attributions = []

    for _, row in X.iterrows():
        row_df = row.to_frame().T

        bd = explainer.predict_parts(
            row_df,
            type="break_down",
            # order=np.array(cols)
        )

        result = bd.result

        pesos = np.zeros(len(cols), dtype=float)
        col_to_idx = {col: idx for idx, col in enumerate(cols)}

        # DALEX incluye normalmente:
        # primera fila = intercepto
        # filas intermedias = variables
        # ultima fila = prediccion final
        for _, result_row in result.iloc[1:-1].iterrows():
            variable_name = result_row['variable_name']
            contribution = result_row['contribution']

            if variable_name in col_to_idx:
                pesos[col_to_idx[variable_name]] = float(contribution)

        attributions.append(pesos)

    return np.asarray(attributions)


def usar_breakdown(
    clf: BaseDetector,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    observaciones: List
):
    
    cols = list(X_train.columns)
    X_obs = X_test.loc[observaciones]

    explainer = dx.Explainer(
        model=clf,
        data=X_train,
        predict_function=_predict_anomaly_score,
        label="PyOD detector",
        verbose=False
    )

    explicaciones = _breakdown_attributions(
        explainer=explainer,
        X=X_obs,
        cols=cols
    )

    return explicaciones


def evaluar_breakdown(
    model: BaseDetector,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test,
    explicaciones: np.ndarray,
    dataset_name: str,
    model_name: str
):
    wrapped_model = QuantusWrapper(model)
    observations = DATASETS[dataset_name]['observations']

    cols = list(X_train.columns)

    explainer = dx.Explainer(
        model=model,
        data=X_train,
        predict_function=_predict_anomaly_score,
        label="PyOD detector",
        verbose=False
    )

    def explain_func(model, inputs, targets=None, **kwargs):
        if isinstance(inputs, torch.Tensor):
            X_np = inputs.detach().cpu().numpy()
        else:
            X_np = np.asarray(inputs)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        X_batch = pd.DataFrame(X_np, columns=cols)

        return _breakdown_attributions(
            explainer=explainer,
            X=X_batch,
            cols=cols
        )
    
    ctx = MetricContext(
        model=wrapped_model,
        X_test=X_test,
        y_test=y_test,
        observations=DATASETS["hydraulic"]["observations"],
        attributions=explicaciones,
        extras={
            "explain_func": explain_func,
            "X_reference": X_train
        },
    )

    metric_results = run_all_metrics(
        ctx,
        selected_metrics=[
            "Complexity",
            "Sparseness",
            "Consistency",
            "FaithfulnessEstimate",
            "MonotonicityCorrelation",
            "Monotonicity",
            "SensitivityN",
            "Sufficiency",
            "Completeness",
            "NonSensitivity",
            "LocalLipschitzEstimate",
            "MaxSensitivity",
            "RelativeInputStability",
            "RelativeOutputStability",
            "AvgSensitivity",
            "Faithfulness",
            "MonotonicityMetric"
        ],
        config="XAI_metrics/config.yaml"
    )
    print(metric_results)

    report_paths = save_metrics_report(
        metric_results=metric_results,
        output_dir=Path("results") / "metric_reports",
        report_name=f"{dataset_name}_{model_name}_breakdown_metrics_report",
        observations=observations,
    )

    print("\nReportes guardados:")
    for fmt, path in report_paths.items():
        print(f"- {fmt}: {path}")

    return metric_results