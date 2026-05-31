# tests/test_base.py
import torch.nn as nn
import pandas as pd
import numpy as np
import pytest

from xai_metrics.base import MetricContext, BaseMetric

def test_metric_context_stores_basic_inputs():
    context = MetricContext(
        model=nn.Identity(),
        X_test=pd.DataFrame({"x1": [1.0, 2.0]}),
        y_test=pd.Series([0, 1]),
        observations=[0, 1],
        attributions=np.array([[0.1], [0.2]]),
        device="cpu"
    )

    assert context.X_test.shape == (2, 1)
    assert context.y_test.tolist() == [0, 1]
    assert context.observations == [0, 1]
    assert context.attributions.shape == (2, 1)
    assert context.device == "cpu"


def test_base_metric_requires_run_implementation(metric_context):
    metric = BaseMetric(context=metric_context)

    with pytest.raises(NotImplementedError):
        metric.run()