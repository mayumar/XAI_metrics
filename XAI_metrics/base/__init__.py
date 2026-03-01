# XAI_metrics/base/__init__.py
from .base import BaseMetric, MetricContext
from .registry import register_metric, METRIC_REGISTRY

__all__ = ['BaseMetric', 'MetricContext', 'register_metric', 'METRIC_REGISTRY']