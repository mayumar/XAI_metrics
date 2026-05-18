# XAI_metrics/runner/runner.py
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

from XAI_metrics.base import MetricContext, MetricSkipped, build_metrics_from_config
from XAI_metrics.metrics import autodiscover_metrics
import XAI_metrics.metrics as metrics_pkg
import XAI_metrics.base.registry as registry

from typing import Iterable, Any, Mapping

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
        "observations",
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
    attributions = np.load(ctx_cfg['attributions_path'])

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
        observations=ctx_cfg["observations"],
        attributions=attributions,
        extras=extras,
    )


def _resolve_metric_selection(
    selected_metrics: Iterable[str] | None,
    configured_metrics: Iterable[str],
    registered_metrics: Iterable[str],
) -> list[str]:
    configured = list(configured_metrics)
    registered = set(registered_metrics)

    selected = configured if selected_metrics is None else list(selected_metrics)

    missing = [name for name in selected if name not in registered]
    if missing:
        raise ValueError(
            f"Metrics not registered: {missing}. "
            f"Available: {sorted(registered)}"
        )

    return selected


def _load_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if isinstance(config, Mapping):
        return dict(config)

    config_path = Path(config)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_all_metrics(
    context: MetricContext | None = None,
    selected_metrics: Iterable[str] | None = None,
    config: Mapping[str, Any] | str | Path | None = None,
    **runtime_kwargs: Any,
) -> dict[str, Any]:
    autodiscover_metrics(metrics_pkg)

    resolved_config = _load_config(config)

    if context is None:
        context = _build_context_from_config(
            resolved_config,
            **runtime_kwargs,
        )

    configured_metrics = [
        metric["name"]
        for metric in resolved_config.get("metrics", [])
    ]

    metric_names = _resolve_metric_selection(
        selected_metrics,
        configured_metrics,
        registry.METRIC_REGISTRY.keys(),
    )
    allowed = set(metric_names)

    filtered_cfg = {
        "metrics": [
            metric
            for metric in resolved_config.get("metrics", [])
            if metric.get("name") in allowed
        ]
    }

    deps = dict(context.extras or {})
    deps.update(runtime_kwargs)

    metrics = build_metrics_from_config(
        filtered_cfg,
        context,
        dependencies=deps,
    )

    out = {"results": {}, "skipped": {}}

    for metric in metrics:
        name = getattr(metric, "NAME", metric.__class__.__name__)

        try:
            out["results"][name] = metric.run()
        except MetricSkipped as exc:
            out["skipped"][name] = str(exc)
            print(str(exc))

    return out