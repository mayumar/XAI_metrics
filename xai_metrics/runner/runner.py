# XAI_metrics/runner/runner.py
from pathlib import Path

from xai_metrics.base import MetricContext, build_metrics_from_config, MetricSkipped
from xai_metrics.metrics import autodiscover_metrics
from xai_metrics.config import ConfigController
from xai_metrics.reporting import build_reports, save_reports
import xai_metrics.metrics as metrics_pkg
import xai_metrics.base.registry as registry

from typing import Iterable, Any, Mapping, Dict, List
import warnings


def _resolve_metric_selection(
    selected_metrics: Iterable[str] | None,
    configured_metrics: Iterable[str],
    registered_metrics: Iterable[str],
) -> List[str]:
    """
    Resolve the final list of metrics to execute.

    The function compares the metrics requested by the user, the metrics
    defined in the configuration, and the metrics registered in the metric
    registry. Metrics that are configured or selected but not registered are
    skipped with a warning. Selected metrics that are not present in the
    configuration are also skipped with a warning.

    Parameters
    ----------
    selected_metrics : Iterable[str] or None
        Optional iterable with the names of the metrics to execute. If
        ``None``, all valid configured metrics are selected.
    configured_metrics : Iterable[str]
        Metric names defined in the configuration file.
    registered_metrics : Iterable[str]
        Metric names available in the metric registry.

    Returns
    -------
    List[str]
        Metric names that are both configured and registered, and that also
        match ``selected_metrics`` when a selection is provided.

    Warns
    -----
    UserWarning
        If configured metrics are not registered, if selected metrics are not
        configured, or if selected metrics are not registered.
    """
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
    """
    Validate metadata for a directly provided metric context.

    When :func:`run_evaluation` receives a ``MetricContext`` directly, the
    metadata cannot be inferred from the configuration and must be provided
    explicitly. The required fields are ``dataset_name``, ``model_name`` and
    ``xai_method_name``.

    Parameters
    ----------
    metadata : Mapping[str, Any] or None
        Metadata associated with the provided metric context.

    Returns
    -------
    Dict[str, Any]
        Validated metadata dictionary.

    Raises
    ------
    ValueError
        If ``metadata`` is ``None`` or if any required metadata field is
        missing or empty.
    """
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
    """
    Run the XAI metric evaluation pipeline.

    The function discovers available metrics, loads the configuration, builds
    one or more metric contexts, resolves the metrics to execute, injects
    runtime dependencies, runs each metric, collects the results, and builds
    summary reports.

    If ``context`` is not provided, contexts are built from ``config`` using
    :class:`~XAI_metrics.config.ConfigController`. If ``context`` is provided
    directly, ``metadata`` must also be provided with ``dataset_name``,
    ``model_name`` and ``xai_method_name``.

    Runtime dependencies can be passed through ``runtime_kwargs``. These values
    are forwarded to metric constructors when their parameter names match. A
    special ``explain_funcs`` dependency can be provided as a mapping from XAI
    method names to explanation functions. In that case, the function matching
    the current ``xai_method_name`` is injected as ``explain_func``.

    Parameters
    ----------
    context : MetricContext or None, optional
        Optional pre-built metric context. If ``None``, contexts are built from
        the configuration.
    metadata : Mapping[str, Any] or None, optional
        Metadata for ``context`` when a context is passed directly. Required
        fields are ``dataset_name``, ``model_name`` and ``xai_method_name``.
    selected_metrics : Iterable[str] or None, optional
        Optional metric names to execute. If ``None``, all valid metrics from
        the configuration are executed.
    config : Mapping[str, Any], str or pathlib.Path, default="config.yaml"
        Configuration source used by :class:`~XAI_metrics.config.ConfigController`.
        It can be a mapping or a path to a YAML file.
    report_output_dir : str, pathlib.Path or None, optional
        Directory where CSV and JSON reports are saved. If ``None``, reports
        are built but not written to disk. The default value is
        ``results/reports``.
    **runtime_kwargs : Any
        Additional runtime dependencies passed to metric constructors. Common
        entries include ``model_loader``, explain functions, custom normalisation
        functions, perturbation functions, similarity functions, and ``explain_funcs``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with three entries:

        - ``"contexts"`` : list
          Raw outputs for each evaluated context, including metadata, metric
          results and skipped metrics.
        - ``"reports"`` : dict
          Report dataframes grouped by dataset and model.
        - ``"report_paths"`` : dict
          Paths to saved CSV and JSON report files. Empty if
          ``report_output_dir`` is ``None``.

    Raises
    ------
    ValueError
        If direct context metadata is missing, if a required explanation
        function is not provided for the current XAI method, or if the
        configuration is invalid.
    """
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

        print(
            f"Evaluating dataset {ctx_metadata['dataset_name']}, "
            f"model {ctx_metadata['model_name']} and "
            f"XAI method {ctx_metadata['xai_method_name']}"
        )

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