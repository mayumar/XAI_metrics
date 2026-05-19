from pathlib import Path
import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer

from XAI_metrics.runner import run_evaluation
from XAI_metrics.base import MetricContext
from XAI_metrics.reporting import save_metrics_report

from utils import QuantusWrapper, load_model
from config import DATASETS

from pyod.models.base import BaseDetector
from typing import List

def _make_lime_explainer(X_train: pd.DataFrame):
    cols = list(X_train.columns)

    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=cols,
        mode="classification",
        random_state=42
    )

    return explainer, cols


def _lime_attributions(
    model,
    explainer: LimeTabularExplainer,
    X: pd.DataFrame,
    cols: List[str]
):
    X_np = np.asarray(X)
    attributions = []

    for row in X_np:
        explanation = explainer.explain_instance(
            data_row=row,
            predict_fn=model.predict_proba,
            num_features=len(cols)
        )

        pesos = np.zeros(len(cols))
        for feature_idx, weight in explanation.as_map()[1]:
            pesos[feature_idx] = float(weight)

        attributions.append(pesos)

    return np.asarray(attributions)



def usar_lime(
    clf: BaseDetector,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    observaciones: List
):

    explainer, cols = _make_lime_explainer(X_train)

    X_obs = X_test.loc[observaciones]
    explicaciones = _lime_attributions(clf, explainer, X_obs, cols)

    return explicaciones


def evaluar_lime(
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

    explainer, cols = _make_lime_explainer(X_train)

    def explain_func(model, inputs, targets=None, **kwargs):
        if hasattr(model, "predict_proba"):
            pyod_model = model
        elif hasattr(model, "model") and hasattr(model.model, "predict_proba"):
            pyod_model = model.model
        else:
            raise AttributeError(
                f"No se encontró predict_proba en {type(model)} ni en model.model"
            )

        return _lime_attributions(
            model=pyod_model,
            explainer=explainer,
            X=inputs,
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

    # metric_results = run_evaluation(
    #     ctx,
    #     selected_metrics=[
    #         "Complexity",
    #         "Sparseness",
    #         "Consistency",
    #         "FaithfulnessEstimate",
    #         "MonotonicityCorrelation",
    #         "Monotonicity",
    #         "SensitivityN",
    #         "Sufficiency",
    #         "Completeness",
    #         "NonSensitivity",
    #         "LocalLipschitzEstimate",
    #         "MaxSensitivity",
    #         "RelativeInputStability",
    #         "RelativeOutputStability"
    #     ],
    #     config="XAI_metrics/config.yaml"
    # )

    metric_results = run_evaluation(
        config="XAI_metrics/config.yaml",
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
            "RelativeOutputStability"
        ],
        model_loader=load_model,
        explain_func=explain_func
    )
    print(metric_results)

    report_paths = save_metrics_report(
        metric_results=metric_results,
        output_dir=Path("results") / "metric_reports",
        report_name=f"{dataset_name}_{model_name}_lime_metrics_report",
        observations=observations,
    )

    print("\nReportes guardados:")
    for fmt, path in report_paths.items():
        print(f"- {fmt}: {path}")

    return metric_results