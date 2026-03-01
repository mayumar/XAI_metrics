# XAI_metrics/base/base.py
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping
import numpy as np
import torch.nn as nn
import pandas as pd

@dataclass(frozen=True)
class MetricContext:
    model: nn.Module
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    observations: Any
    attributions: np.ndarray
    extras: Dict[str, Any] = field(default_factory=dict)

class BaseMetric:
    NAME: str = 'metric'

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        self.context = context
        self.params = dict(params or {})
    
    def run(self):
        raise NotImplementedError("This class does not implement a run method")