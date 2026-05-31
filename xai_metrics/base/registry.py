# XAI_metrics/base/registry.py
import inspect
from typing import Dict, Type, Mapping, Any, List
from xai_metrics.base import BaseMetric, MetricContext

METRIC_REGISTRY = {}

def register_metric(cls: Type[BaseMetric]) -> Type[BaseMetric]:
    """
    Register a metric class in the global metric registry.

    The metric is registered using its ``NAME`` attribute. If the class does not
    define ``NAME``, the class name is used instead.

    Parameters
    ----------
    cls : Type[BaseMetric]
        Metric class to register.

    Returns
    -------
    Type[BaseMetric]
        The same metric class passed as input. This allows the function to be
        used as a decorator.

    Raises
    ------
    ValueError
        If another metric with the same name is already registered.

    Examples
    --------
    >>> @register_metric
    ... class MyMetric(BaseMetric):
    ...     NAME = "my_metric"
    ...     def run(self):
    ...         return 0.0
    """
    name = getattr(cls, 'NAME', cls.__name__)

    if name in METRIC_REGISTRY:
        raise ValueError(f"Duplicate metric name registered: {name}")
    
    METRIC_REGISTRY[name] = cls

    return cls


def list_metrics() -> List[str]:
    """
    Return the names of the registered metrics.

    Returns
    -------
    List[str]
        Sorted list containing the names of all metrics currently registered in
        ``METRIC_REGISTRY``.
    """
    return sorted(METRIC_REGISTRY.keys())


def build_metrics_from_config(
    metrics_cfg: List[Mapping[str, Any]],
    context: MetricContext,
    dependencies: Mapping[str, Any] | None = None
) -> List[BaseMetric]:
    """
    Build metric instances from a configuration list.

    Each metric configuration must contain a ``name`` field matching a
    registered metric. It may also contain a ``params`` field with
    metric-specific parameters. Additional dependencies are injected only if
    their names are accepted by the metric class constructor.

    Parameters
    ----------
    metrics_cfg : List[Mapping[str, Any]]
        List of metric configuration dictionaries. Each dictionary must contain
        the metric ``name`` and may contain ``params``.
    context : MetricContext
        Shared metric evaluation context passed to every metric instance.
    dependencies : Mapping[str, Any] or None, optional
        Optional dependency objects that can be injected into metric
        constructors. Only dependencies whose names appear in the constructor
        signature are passed.

    Returns
    -------
    List[BaseMetric]
        List of instantiated metric objects.

    Raises
    ------
    ValueError
        If a metric name from ``metrics_cfg`` is not registered.
    KeyError
        If a metric configuration does not contain the required ``name`` field.
    """
    dependencies = dict(dependencies or {})
    instances: List[BaseMetric] = []

    for metric_cfg in metrics_cfg:
        name = metric_cfg["name"]
        params = metric_cfg.get("params")

        if name not in METRIC_REGISTRY:
            raise ValueError(
                f"Metric '{name}' not registered. "
                f"Available: {list(METRIC_REGISTRY.keys())}"
            )

        metric_cls = METRIC_REGISTRY[name]
        sig = inspect.signature(metric_cls.__init__)
        allowed = set(sig.parameters.keys())
        kwargs = {k: v for k, v in dependencies.items() if k in allowed}

        metric = metric_cls(context=context, params=params, **kwargs)
        instances.append(metric)

    return instances