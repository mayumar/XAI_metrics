import pandas as pd
import numpy as np
import shap
import torch
from pathlib import Path

from utils import QuantusWrapper, load_model
from config import DATASETS

from xai_metrics.runner import run_evaluation
from xai_metrics.base import MetricContext

from pyod.models.base import BaseDetector
from typing import List


def _shap_attributions(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    def f(X):
        if hasattr(model, "decision_function"):
            return model.decision_function(X)
        return model

    # Importancia de la variable en el score de anomalia
    background_shap = shap.sample(X_train, 100, random_state=0)
    explainer = shap.Explainer(f, background_shap)
    shap_values = explainer(X_test)

    return np.abs(shap_values.values)


def make_shap_local_explain_func(X_train: pd.DataFrame):
    def explain_func(model, inputs, targets=None, **kwargs):
        if isinstance(inputs, torch.Tensor):
            X_np = inputs.detach().cpu().numpy()
        else:
            X_np = np.asarray(inputs)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        X_batch = pd.DataFrame(X_np, columns=X_train.columns)

        if hasattr(model, "decision_function"):
            pyod_model = model
        elif hasattr(model, "model") and hasattr(model.model, "decision_function"):
            pyod_model = model.model
        else:
            raise AttributeError(
                f"No se encontró decision_function en {type(model)} ni en model.model"
            )

        return _shap_attributions(
            model=pyod_model,
            X_train=X_train,
            X_test=X_batch
        )
    
    return explain_func


def usar_shap_local(
    clf: BaseDetector,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    observaciones: List
):
    X_obs = X_test.loc[observaciones]

    explicaciones = _shap_attributions(
        model=clf,
        X_train=X_train,
        X_test=X_obs
    )

    return explicaciones


def usar_shap_global(
    clf: BaseDetector,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
):
    explicaciones = _shap_attributions(
        model=clf,
        X_train=X_train,
        X_test=X_test
    )

    explicacion_global = np.mean(explicaciones, axis=0)

    return explicacion_global


def evaluar_shap_local(
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

    def explain_func(model, inputs, targets=None, **kwargs):
        if isinstance(inputs, torch.Tensor):
            X_np = inputs.detach().cpu().numpy()
        else:
            X_np = np.asarray(inputs)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        X_batch = pd.DataFrame(X_np, columns=X_train.columns)

        if hasattr(model, "decision_function"):
            pyod_model = model
        elif hasattr(model, "model") and hasattr(model.model, "decision_function"):
            pyod_model = model.model
        else:
            raise AttributeError(
                f"No se encontró decision_function en {type(model)} ni en model.model"
            )

        return _shap_attributions(
            model=pyod_model,
            X_train=X_train,
            X_test=X_batch
        )
    
    ctx = MetricContext(
        model=wrapped_model,
        X_test=X_test,
        y_test=y_test,
        observations=DATASETS["hydraulic"]["observations"],
        attributions=explicaciones,
    )

    metadada = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "xai_method_name": "SHAP_local"
    }

    metric_results = run_evaluation(
        ctx,
        metadata=metadada,
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
        explain_funcs={"SHAP_local": explain_func}
    )
    print(metric_results)

    return metric_results


def evaluar_shap_global(
    model: BaseDetector,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test,
    explicacion_global: np.ndarray,
    dataset_name: str,
    model_name: str
):
    wrapped_model = QuantusWrapper(model)

    explicacion_global = np.asarray(explicacion_global, dtype=float)

    attributions = np.tile(
        explicacion_global,
        (len(X_test), 1)
    )

    def explain_func(model_wrapper, inputs, target=None, **kwargs):
        if isinstance(inputs, torch.Tensor):
            X_np = inputs.detach().cpu().numpy()
        else:
            X_np = np.asarray(inputs)

        if X_np.ndim == 1:
            X_np = X_np.reshape(1, -1)

        return np.tile(
            explicacion_global,
            (X_np.shape[0], 1)
        )

    ctx = MetricContext(
        model=wrapped_model,
        X_test=X_test,
        y_test=y_test,
        observations=None,
        attributions=attributions
    )

    metadada = {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "xai_method_name": "SHAP_global"
    }

    metric_results = run_evaluation(
        ctx,
        metadata=metadada,
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
        explain_funcs={"SHAP_global": explain_func}
    )

    return metric_results