# tests/test_base.py
import pytest

from dataclasses import FrozenInstanceError

from XAI_metrics.base import MetricContext, BaseMetric

def test_metric_context_exposes_expected_fields(metric_context: MetricContext):
    assert list(metric_context.X_test.columns) == ["f1", "f2"]
    assert list(metric_context.y_test.index) == [10, 11]
    assert metric_context.observations == [10, 11]
    assert metric_context.attributions.shape == (2, 2)

def test_metric_context_is_frozen(metric_context: MetricContext):
    with pytest.raises(FrozenInstanceError):
        metric_context.model = None # type: ignore

def test_base_metric_stores_context_and_params(metric_context: MetricContext):
    metric = BaseMetric(metric_context, params={"flag": True})
    assert metric.context is metric_context
    assert metric.params == {"flag": True}

def test_base_metric_run_raises_not_implemented(metric_context: MetricContext):
    metric = BaseMetric(metric_context)
    with pytest.raises(NotImplementedError):
        metric.run()