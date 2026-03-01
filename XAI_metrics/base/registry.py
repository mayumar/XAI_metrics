# XAI_metrics/base/registry.py
from typing import Dict, Type, Mapping, Any, List
from XAI_metrics.base import BaseMetric, MetricContext

METRIC_REGISTRY: Dict[str, BaseMetric] = {}

def register_metric(cls: Type[BaseMetric]) -> Type[BaseMetric]:
    name = getattr(cls, 'NAME', cls.__name__)

    if name in METRIC_REGISTRY:
        raise ValueError(f"Duplicate metric name registered: {name}")
    
    METRIC_REGISTRY[name] = cls

    return cls

def list_metrics() -> list[str]:
    return sorted(METRIC_REGISTRY.keys())

def build_metrics_from_config(
    config: Mapping[str, Any],
    context: MetricContext,
    dependencies: Mapping[str, Any] | None = None
) -> List[BaseMetric]:
    
    dependencies = dict(dependencies or {})
    metrics_cfg = config.get("metrics", [])
    instances = []

    for metric_cfg in metrics_cfg:
        name = metric_cfg["name"]
        params = metric_cfg.get("params")

        if name not in METRIC_REGISTRY:
            raise ValueError(
                f"Metric '{name}' not registered."
                f"Available: {list(METRIC_REGISTRY.keys())}"
            )
        
        metric_cls = METRIC_REGISTRY[name]

        try:
            metric = metric_cls(context=context, params=params, **dependencies)
        except TypeError:
            metric = metric_cls(context=context, params=params)

        instances.append(metric)