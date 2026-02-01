from dataclasses import dataclass, field
from typing import Any, Dict
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

    def __init__(self, contextParams: MetricContext):
        self.contextParams = contextParams
    
    def run(self):
        raise NotImplementedError("This class does not implement a run method")