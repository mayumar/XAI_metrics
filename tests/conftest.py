# tests/conftest.py
import pytest
import torch.nn as nn
import pandas as pd
import numpy as np

from xai_metrics.base import BaseMetric, METRIC_REGISTRY, MetricContext

class DummyMetric(BaseMetric):
    NAME = "dummy"

    def run(self):
        return self.params.get("value", 1.0)
    

@pytest.fixture
def dummy_metric_class():
    return DummyMetric


@pytest.fixture
def clean_registry():
    old_registry = dict(METRIC_REGISTRY)
    METRIC_REGISTRY.clear()

    yield

    METRIC_REGISTRY.clear()
    METRIC_REGISTRY.update(old_registry)


@pytest.fixture
def metric_context():
    return MetricContext(
        model=nn.Identity(),
        X_test=pd.DataFrame({"x1": [1.0]}),
        y_test=pd.Series([0]),
        observations=[0],
        attributions=np.array([[0.5]])
    )