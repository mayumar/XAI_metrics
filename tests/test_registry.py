# tests/test_registry.py
from typing import Any, Mapping

import pytest

from XAI_metrics.base import (
    BaseMetric,
    METRIC_REGISTRY,
    register_metric,
    list_metrics,
    MetricContext,
    build_metrics_from_config
)

def test_register_metric_adds_class_to_registry():
    class TempMetric(BaseMetric):
        NAME = "TempMetric"

        def run(self):
            return "ok"
        
    METRIC_REGISTRY.pop("TempMetric", None)

    register_metric(TempMetric)

    assert METRIC_REGISTRY["TempMetric"] is TempMetric

    METRIC_REGISTRY.pop("TempMetric", None)

def test_register_metric_raises_for_duplicate_name():
    class TempMetricDuplicateA(BaseMetric):
        NAME = "TempMetricDuplicate"

        def run(self):
            return "a"
        
    class TempMetricDuplicateB(BaseMetric):
        NAME = "TempMetricDuplicate"

        def run(self):
            return "b"
        
    METRIC_REGISTRY.pop("TempMetricDuplicate", None)
    register_metric(TempMetricDuplicateA)

    with pytest.raises(ValueError, match="Duplicate metric name"):
        register_metric(TempMetricDuplicateB)

    METRIC_REGISTRY.pop("TempMetricDuplicate", None)

def test_list_metrics_returns_sorted_names():
    class MetricZ(BaseMetric):
        NAME = "ZZMetric"

        def run(self):
            return None
        
    class MetricA(BaseMetric):
        NAME = "AAMetric"

        def run(self):
            return None
        
    METRIC_REGISTRY.pop("ZZMetric", None)
    METRIC_REGISTRY.pop("AAMetric", None)

    register_metric(MetricZ)
    register_metric(MetricA)

    names = list_metrics()

    assert "AAMetric" in names
    assert "ZZMetric" in names
    assert names.index("AAMetric") < names.index("ZZMetric")

    METRIC_REGISTRY.pop("ZZMetric", None)
    METRIC_REGISTRY.pop("AAMetric", None)

def test_build_metrics_from_config_creates_instances(metric_context: MetricContext):
    class BuildMetric(BaseMetric):
        NAME = "BuildMetric"

        def run(self):
            return "ok"
        
    METRIC_REGISTRY.pop("BuildMetric", None)
    register_metric(BuildMetric)

    config = {"metrics": [{"name": "BuildMetric", "params": {"x": 1}}]}
    metrics = build_metrics_from_config(config, metric_context)

    assert len(metrics) == 1
    assert isinstance(metrics[0], BuildMetric)
    assert metrics[0].params == {"x": 1}

    METRIC_REGISTRY.pop("BuildMetric", None)

def test_build_metrics_from_config_raises_for_unknown_metric(metric_context: MetricContext):
    config = {"metrics": [{"name": "DoesNotExist"}]}

    with pytest.raises(ValueError, match="not registered"):
        build_metrics_from_config(config, metric_context)

def test_build_metrics_from_config_passes_only_supported_dependencies(metric_context: MetricContext):
    class DependencyMetric(BaseMetric):
        NAME = "DependencyMetric"

        def __init__(self, context, params = None, explain_func = None):
            super().__init__(context, params)
            self.explain_func = explain_func
        
        def run(self):
            return self.explain_func is not None
        
    METRIC_REGISTRY.pop("DependencyMetric", None)
    register_metric(DependencyMetric)

    config = {"metrics": [{"name": "DependencyMetric"}]}
    deps = {"explain_func": lambda *args, **kwargs: None, "unused": "ignored"}

    metrics = build_metrics_from_config(config, metric_context, dependencies=deps)

    assert len(metrics) == 1
    assert metrics[0].explain_func is not None # type: ignore
    assert not hasattr(metrics[0], "unused")

    METRIC_REGISTRY.pop("DependencyMetric", None)