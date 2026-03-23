# tests/conftest.py
import pandas as pd
import numpy as np
import pytest

from XAI_metrics.base import MetricContext

class DummyModel:
    def __init__(self):
        self.mode = None
    def train(self):
        self.mode = "train"
    def eval(self):
        self.mode = "eval"

@pytest.fixture
def metric_context():
    X = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=[10, 11],
        columns=["f1", "f2"]
    )
    y = pd.Series([0, 1], index=[10, 11])
    a = np.array([[0.1, 0.9], [0.3, 0.7]], dtype=float)

    return MetricContext(
        model=DummyModel(),
        X_test=X,
        y_test=y,
        observations=[10, 11],
        attributions=a,
        extras={}
    )

@pytest.fixture
def explain_func():
    def _explain(mode, inputs, targets=None, **kwargs):
        arr = np.asarray(inputs, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return np.ones_like(arr, dtype=float)
    
    return _explain