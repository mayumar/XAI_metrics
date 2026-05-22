# XAI_metrics/runner/runner.py
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from XAI_metrics.base import MetricContext
from XAI_metrics.metrics import autodiscover_metrics
from XAI_metrics.config import ConfigController
import XAI_metrics.metrics as metrics_pkg
import XAI_metrics.base.registry as registry

from typing import Iterable, Any, Mapping, Dict
import warnings

def _normalize_selection(
    selected_metrics: Iterable[str] | None,
    available_metrics: Iterable[str],
) -> list[str]:
    available_list = list(available_metrics)

    if selected_metrics is None:
        return available_list

    selected = list(selected_metrics)
    available_set = set(available_list)

    missing = [name for name in selected if name not in available_set]
    if missing:
        raise ValueError(
            f"Metrics not registered: {missing}. "
            f"Available: {sorted(available_list)}"
        )

    return selected


def _build_context_from_config(
    config: Mapping[str, Any],
    **runtime_kwargs: Any
) -> MetricContext:
    ctx_cfg = config.get("context")
    if not ctx_cfg:
        raise ValueError("No MetricContext provided and config has no 'context' section.")
    
    required = [
        "model_path",
        "X_test_path",
        "y_test_path",
        "attributions_path"
    ]
    missing = [key for key in required if key not in ctx_cfg]
    if missing:
        raise ValueError(f"Missing context config fields: {missing}")
    
    model_loader = runtime_kwargs.get("model_loader")
    if model_loader is None:
        raise ValueError("Building MetricContext from config requires 'model_loader' to be passed at runtime.")
    
    model = model_loader(ctx_cfg['model_path'])
    X_test = pd.read_csv(ctx_cfg['X_test_path'], index_col=0)
    y_test = pd.read_csv(ctx_cfg['y_test_path'], index_col=0).squeeze('columns')
    attributions_df = pd.read_csv(ctx_cfg["attributions_path"], index_col=0)

    observations = attributions_df.index.tolist()
    attributions = attributions_df.to_numpy()

    extras: dict[str, Any] = {}

    if "X_reference_path" in ctx_cfg:
        extras["X_reference"] = pd.read_csv(
            ctx_cfg["X_reference_path"],
            index_col=0,
        )

    return MetricContext(
        model=model,
        X_test=X_test,
        y_test=y_test,
        observations=observations,
        attributions=attributions,
        extras=extras,
    )


def _resolve_metric_selection(
    selected_metrics: Iterable[str] | None,
    configured_metrics: Iterable[str],
    registered_metrics: Iterable[str],
) -> list[str]:
    configured = set(list(configured_metrics))
    registered = set(registered_metrics)

    configured_not_registered = [
        name for name in configured
        if name not in registered
    ]

    if configured_not_registered:
        warnings.warn(
            "Configured metrics not registered and will be skipped: "
            f"{configured_not_registered}. "
            f"Available: {sorted(registered)}",
            UserWarning,
            stacklevel=2,
        )

    valid_configured = [
        name for name in configured
        if name in registered
    ]

    if selected_metrics is None:
        return valid_configured
    
    selected = list(selected_metrics)

    selected_not_configured = [
        name for name in selected
        if name not in configured
    ]

    if selected_not_configured:
        warnings.warn(
            "Selected metrics not present in config and will be skipped: "
            f"{selected_not_configured}. "
            f"Configured: {configured}",
            UserWarning,
            stacklevel=2,
        )

    selected_not_registered = [
        name for name in selected
        if name not in registered
    ]

    if selected_not_registered:
        warnings.warn(
            "Selected metrics not registered and will be skipped: "
            f"{selected_not_registered}. "
            f"Available: {sorted(registered)}",
            UserWarning,
            stacklevel=2,
        )

    return [
        name for name in selected
        if name in configured and name in registered
    ]


def _validate_context_metadata(metadata: Mapping[str, Any] | None) -> Dict[str, Any]:
    required = ["dataset_name", "model_name", "xai_method_name"]

    if metadata is None:
        raise ValueError(
            f"When passing a MetricContext directly, metadata must also be provided with fields: {required}."
        )

    missing = [key for key in required if not metadata.get(key)]
    if missing:
        raise ValueError(
            "Context metadata is missing required fields: "
            f"{missing}. Required fields: {required}."
        )

    return dict(metadata)


def run_evaluation(
    context: MetricContext | None = None,
    metadata: Mapping[str, Any] | None = None,
    selected_metrics: Iterable[str] | None = None,
    config: Mapping[str, Any] | str | Path = "config.yaml",
    **runtime_kwargs: Any,
) -> Dict[str, Any]:
    autodiscover_metrics(metrics_pkg)

    config_controller = ConfigController(
        config=config,
        model_loader=runtime_kwargs.get("model_loader")
    )

    if context is None:
        context_list = config_controller.build_contexts()
    else:
        context_list = [
            (context, _validate_context_metadata(metadata))
        ]

    metrics_config = config_controller.get_metrics_config()
    configured_metrics = [
        metric['name']
        for metric in metrics_config
    ]

    metric_names = _resolve_metric_selection(
        selected_metrics,
        configured_metrics,
        registry.METRIC_REGISTRY.keys(),
    )
    allowed = set(metric_names)

    filtered_cfg = [
        metric
        for metric in metrics_config
        if metric.get("name") in allowed
    ]

    print(context_list)
    #=====================

    # deps = dict(context.extras or {})
    # deps.update(runtime_kwargs)

    # metrics = build_metrics_from_config(
    #     filtered_cfg,
    #     context,
    #     dependencies=deps,
    # )

    # out = {"results": {}, "skipped": {}}

    # for metric in metrics:
    #     name = getattr(metric, "NAME", metric.__class__.__name__)

    #     try:
    #         out["results"][name] = metric.run()
    #     except MetricSkipped as exc:
    #         out["skipped"][name] = str(exc)
    #         print(str(exc))

    # return out