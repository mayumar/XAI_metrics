# XAI_metrics/runner/runner.py
from pathlib import Path

from XAI_metrics.base import MetricContext, build_metrics_from_config, MetricSkipped
from XAI_metrics.metrics import autodiscover_metrics
from XAI_metrics.config import ConfigController
from XAI_metrics.reporting import build_reports, save_reports
import XAI_metrics.metrics as metrics_pkg
import XAI_metrics.base.registry as registry

from typing import Iterable, Any, Mapping, Dict, List
import warnings


def _resolve_metric_selection(
    selected_metrics: Iterable[str] | None,
    configured_metrics: Iterable[str],
    registered_metrics: Iterable[str],
) -> List[str]:
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
    report_output_dir: str | Path | None = Path("results") / "reports",
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

    context_outputs = []

    for ctx, ctx_metadata in context_list:

        print(f"Evaluando el dataset {ctx_metadata['dataset_name']}, con el modelo {ctx_metadata['model_name']} y el método {ctx_metadata['xai_method_name']}")

        deps = {}

        runtime_deps = dict(runtime_kwargs)
        explain_funcs = runtime_deps.pop("explain_funcs", None)

        deps.update(runtime_deps)

        if explain_funcs is not None:
            xai_method_name = str(ctx_metadata['xai_method_name']).lower()
            explain_func = explain_funcs.get(xai_method_name)

            if explain_func is None:
                raise ValueError(
                    f"No explain_func provided for XAI method '{xai_method_name}'. "
                    f"Available: {list(explain_funcs.keys())}"
                )

            deps['explain_func'] = explain_func

        metrics = build_metrics_from_config(
            filtered_cfg,
            ctx,
            dependencies=deps,
        )
        
        out = {"metadata": ctx_metadata, "results": {}, "skipped": {}}

        for metric in metrics:
            name = getattr(metric, "NAME", metric.__class__.__name__)

            try:
                out["results"][name] = metric.run()
            except MetricSkipped as exc:
                out["skipped"][name] = str(exc)
                print(str(exc))

        context_outputs.append(out)
    
    reports = build_reports(context_outputs)
    report_paths = {}

    if report_output_dir is not None:
        report_paths = save_reports(
            reports=reports,
            output_dir=report_output_dir,
        )

    return {
        "contexts": context_outputs,
        "reports": reports,
        "report_paths": report_paths,
    }