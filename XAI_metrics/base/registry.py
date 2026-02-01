# XAI_metrics/base/registry.py
from typing import Dict, Type
from XAI_metrics.base import BaseMetric

METRIC_REGISTRY: Dict[str, BaseMetric] = {}

def register_metric(cls: Type[BaseMetric]) -> Type[BaseMetric]:
    name = getattr(cls, 'NAME', cls.__name__)

    if name in METRIC_REGISTRY:
        raise ValueError(f"Duplicate metric name registered: {name}")
    
    METRIC_REGISTRY[name] = cls

    return cls

def list_metrics() -> list[str]:
    return sorted(METRIC_REGISTRY.keys())