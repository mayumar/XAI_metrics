# XAI_metrics/runner/runner.py
from pathlib import Path

from xai_metrics.base import MetricContext, build_metrics_from_config, MetricSkipped, ExplainerContext, build_explainers_from_config, METRIC_REGISTRY, EXPLAINER_REGISTRY
from xai_metrics.metrics import autodiscover_metrics
from xai_metrics.config import ConfigController
from xai_metrics.reporting import build_reports, save_reports, save_attributions
from xai_metrics.explainers import autodiscover_explainers
import xai_metrics.metrics as metrics_pkg
import xai_metrics.explainers as explainers_pkg

from typing import Iterable, Any, Mapping, Dict, List
import warnings


def _resolve_explainer_selection(
    selected_explainers: Iterable[str] | None,
    configured_explainers: Iterable[str],
    registered_explainers: Iterable[str]
) -> List[str]:
    configured = list(configured_explainers)
    registered = set(registered_explainers)

    configured_not_registered = [
        name for name in configured
        if name not in registered
    ]

    if configured_not_registered:
        warnings.warn(
            "Configured explainers not registered and will be skipped: "
            f"{configured_not_registered}. "
            f"Available: {sorted(registered)}",
            UserWarning,
            stacklevel=2,
        )

    valid_configured = [
        name for name in configured
        if name in registered
    ]

    if selected_explainers is None:
        return valid_configured
    
    selected = list(selected_explainers)

    selected_not_configured = [
        name for name in selected
        if name not in configured
    ]

    if selected_not_configured:
        warnings.warn(
            "Selected explainers not present in config and will be skipped: "
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
            "Selected explainers not registered and will be skipped: "
            f"{selected_not_registered}. "
            f"Available: {sorted(registered)}",
            UserWarning,
            stacklevel=2,
        )

    return [
        name for name in configured
        if name in selected and name in registered
    ]


def run_explanation(
    context: ExplainerContext | None = None,
    selected_explainers: Iterable[str] | None = None,
    config: Mapping[str, Any] | str | Path = "config.yaml",
    attribution_output_dir: str | Path | None = Path("results") / "attributions",
    **runtime_kwargs: Any
) -> Dict[str, Any]:
    autodiscover_explainers(explainers_pkg)

    config_controller = ConfigController(
        config=config,
        model_loader=runtime_kwargs.get("model_loader")
    )

    ctx_cfg = dict(config_controller.config.get("context") or {})

    if not ctx_cfg:
        raise ValueError("Config must include a 'context' section.")

    if context is None:
        context = config_controller.build_explainers_context(ctx_cfg)

    explainers_config = config_controller.get_explainers_config()
    configured_explainers = [
        explainer['name']
        for explainer in explainers_config
    ]

    explainer_names = _resolve_explainer_selection(
        selected_explainers,
        configured_explainers,
        EXPLAINER_REGISTRY.keys()
    )
    allowed = set(explainer_names)

    filtered_cfg = [
        explainer
        for explainer in explainers_config
        if explainer.get("name") in allowed
    ]

    explainer_params_by_name = {
        str(explainer_cfg['name']): dict(explainer_cfg.get("params") or {})
        for explainer_cfg in filtered_cfg
    }

    metadata = {
        "dataset_name": ctx_cfg.get("dataset_name"),
        "model_name": ctx_cfg.get("model_name")
    }

    missing_metadata = [
        key for key, value in metadata.items()
        if not value
    ]

    if missing_metadata:
        raise ValueError(
            "Explanation metadata is missing required fields: "
            f"{missing_metadata}. Required fields: ['dataset_name', 'model_name']."
        )

    if context.model is None:
        raise ValueError("ExplainerContext must include a model.")

    if context.X_batch is None:
        raise ValueError("ExplainerContext must include X_batch.")

    print("Explaining")

    explainers = build_explainers_from_config(
        filtered_cfg,
        context
    )

    out = {
        "metadata": metadata,
        "attributions": {},
        "explainer_params": {},
        "skipped": {}
    }

    for explainer in explainers:
        name = getattr(explainer, "NAME", explainer.__class__.__name__)

        try:
            out['attributions'][name] = explainer.explain(
                model=context.model,
                inputs=context.X_batch,
                targets=context.y_batch,
                device=context.device
            )

            out['explainer_params'][name] = explainer_params_by_name.get(name, {})

        except Exception as exc:
            out['skipped'][name] = str(exc)
            print(str(exc))

    context_outputs = [out]

    attribution_paths = {}

    if attribution_output_dir is not None:
        attribution_context_outputs = []

        for explainer_name, attributions in out['attributions'].items():
            attribution_context_outputs.append(
                {
                    "metadata": {
                        **metadata,
                        "xai_method_name": explainer_name
                    },
                    "attributions": attributions,
                    "observations": context.X_batch.index,
                    "features": context.X_batch.columns
                }
            )

        attribution_paths = save_attributions(
            context_outputs=attribution_context_outputs,
            output_dir=attribution_output_dir
        )

    return {
        "contexts": context_outputs,
        "attribution_paths": attribution_paths
    }



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
    configured = list(configured_metrics)
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
        name for name in configured
        if name in selected and name in registered
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

    The function discovers all metric modules, loads the experiment
    configuration, builds or accepts one or more metric contexts, resolves the
    metrics that can be executed, injects runtime dependencies and evaluates
    each metric.

    When ``context`` is ``None``, contexts are built from ``config`` through
    :class:`~xai_metrics.config.ConfigController`. When a context is supplied
    directly, ``metadata`` must also be provided.

    Runtime keyword arguments are forwarded as dependencies to metric
    constructors when their names match constructor parameters. The special
    ``model_loader`` argument is also passed to
    :class:`~xai_metrics.config.ConfigController`.

    The special ``explain_funcs`` argument may contain a mapping from XAI
    method names to explanation functions. Method names are matched
    case-insensitively. The function corresponding to the current context is
    injected into compatible metrics under the ``explain_func`` dependency.

    Successfully evaluated metrics are stored together with the parameters
    defined for them in the configuration. Metrics that raise
    :class:`~xai_metrics.base.MetricSkipped` are recorded separately and do not
    stop the remaining evaluations.

    Finally, the collected outputs are transformed into report dataframes. If
    ``report_output_dir`` is not ``None``, the reports are also saved to disk.

    Parameters
    ----------
    context : MetricContext or None, optional
        Pre-built metric evaluation context. If ``None``, one or more contexts
        are constructed from ``config``.
    metadata : Mapping[str, Any] or None, optional
        Metadata associated with a directly supplied ``context``. It must
        contain ``dataset_name``, ``model_name`` and ``xai_method_name``.
        Ignored when contexts are built from the configuration.
    selected_metrics : Iterable[str] or None, optional
        Metric names to execute. Only metrics that are also configured and
        registered are evaluated. If ``None``, all configured and registered
        metrics are evaluated.
    config : Mapping[str, Any], str or pathlib.Path, default="config.yaml"
        Experiment configuration source. It may be a mapping or the path to a
        YAML configuration file.
    report_output_dir : str, pathlib.Path or None, optional
        Directory where the generated reports are saved. If ``None``, reports
        are built but are not written to disk. The default directory is
        ``results/reports``.
    **runtime_kwargs : Any
        Runtime dependencies made available to metric constructors. Supported
        values depend on the selected metrics and may include custom similarity,
        normalisation, perturbation or explanation functions.

        Two arguments receive special handling:

        - ``model_loader`` : Callable, optional
          Function used by the configuration controller to load models.
        - ``explain_funcs`` : Mapping[str, Callable], optional
          Mapping from XAI method names to explanation functions. Method names
          are matched case-insensitively.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the following entries:

        - ``"contexts"`` : List[Dict[str, Any]]
          Raw output for each evaluated context. Every context output contains
          ``metadata``, ``results``, ``metric_params`` and ``skipped``.
        - ``"reports"`` : Dict[str, Dict[str, pandas.DataFrame]]
          Report dataframes grouped first by dataset name and then by model
          name.
        - ``"report_paths"`` : Dict[str, str]
          Paths of the saved report files. The dictionary is empty when
          ``report_output_dir`` is ``None``.

    Raises
    ------
    ValueError
        If metadata required for a directly supplied context is missing, if no
        explanation function is available for the current XAI method, or if
        the configuration or report data is invalid.
    TypeError
        If context construction loads an object that is not a supported model
        type.
    RuntimeError
        If context construction requests a device that is not available.

    Warns
    -----
    UserWarning
        If configured or selected metrics cannot be executed because they are
        missing from the configuration or metric registry.
    """
    autodiscover_metrics(metrics_pkg)

    config_controller = ConfigController(
        config=config,
        model_loader=runtime_kwargs.get("model_loader")
    )

    if context is None:
        context_list = config_controller.build_metric_contexts()
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
        METRIC_REGISTRY.keys(),
    )
    allowed = set(metric_names)

    filtered_cfg = [
        metric
        for metric in metrics_config
        if metric.get("name") in allowed
    ]

    metric_params_by_name = {
        str(metric_cfg['name']): dict(metric_cfg.get("params") or {})
        for metric_cfg in filtered_cfg
    }

    context_outputs = []
    report_paths = {}

    for ctx, ctx_metadata in context_list:

        print(
            f"\n{"=" * 60}\n"
            "Running XAI evaluation\n"
            f"  Dataset    : {ctx_metadata['dataset_name']}\n"
            f"  Model      : {ctx_metadata['model_name']}\n"
            f"  XAI method : {ctx_metadata['xai_method_name']}\n"
            f"{"=" * 60}"
        )

        deps = {}

        runtime_deps = dict(runtime_kwargs)
        explain_funcs = runtime_deps.pop("explain_funcs", None)

        deps.update(runtime_deps)

        if explain_funcs is not None:
            dataset_name = str(ctx_metadata["dataset_name"]).lower()
            xai_method_name = str(ctx_metadata["xai_method_name"]).lower()

            normalized = {
                str(key).lower(): value
                for key, value in explain_funcs.items()
            }

            # Si los valores son funciones, el mapping es global.
            is_global_mapping = all(callable(value) for value in normalized.values())

            if is_global_mapping:
                available_funcs = normalized
            else:
                dataset_mapping = normalized.get(dataset_name)

                if dataset_mapping is None:
                    raise ValueError(
                        f"No explain_funcs provided for dataset '{dataset_name}'. "
                        f"Available datasets: {list(normalized)}"
                    )

                available_funcs = {
                    str(method).lower(): func
                    for method, func in dataset_mapping.items()
                }

            explain_func = available_funcs.get(xai_method_name)

            # Permite que "shap_local" encuentre una función registrada como "shap".
            if explain_func is None and xai_method_name == "shap_local":
                explain_func = available_funcs.get("shap")

            if explain_func is None:
                raise ValueError(
                    f"No explain_func provided for dataset '{dataset_name}' "
                    f"and XAI method '{xai_method_name}'. "
                    f"Available methods: {list(available_funcs)}"
                )

            deps["explain_func"] = explain_func

        metrics = build_metrics_from_config(
            filtered_cfg,
            ctx,
            dependencies=deps,
        )
        
        out = {
            "metadata": ctx_metadata,
            "results": {},
            "metric_params": {},
            "skipped": {}
        }

        for metric in metrics:
            name = getattr(metric, "NAME", metric.__class__.__name__)

            try:
                out['results'][name] = metric.run()
                out['metric_params'][name] = metric_params_by_name.get(name, {})
            except MetricSkipped as exc:
                out['skipped'][name] = str(exc)
                print(str(exc))

        context_outputs.append(out)
    
        if report_output_dir is not None:
            report_paths = save_reports(
                reports=build_reports(context_outputs),
                output_dir=report_output_dir,
            )

    reports = build_reports(context_outputs)

    return {
        "contexts": context_outputs,
        "reports": reports,
        "report_paths": report_paths,
    }