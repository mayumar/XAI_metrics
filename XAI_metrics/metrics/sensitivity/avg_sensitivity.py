import quantus
from base import BaseMetric, MetricContext
from typing import Callable, Any
import torch.nn as nn
import numpy as np
NR_SAMPLES = 20

type ExplainFunc = Callable[[nn.Module, Any, Any | None], np.ndarray]

class AvgSensitivity(BaseMetric):
    NAME = 'AvgSensitivity'

    def __init__(
        self,
        contextParams: MetricContext,
        explain_func: ExplainFunc
    ):
        super().__init__(contextParams)
        self.explain_func = explain_func
    
    def run(self):
        params = self.contextParams

        params.model.train()

        results = quantus.AvgSensitivity(
            abs=True,
            normalise=False,
            nr_samples=NR_SAMPLES
        )(
            model=params.model,
            x_batch=params.X_test.loc[params.observations].values,
            y_batch=params.y_test.loc[params.observations].values,
            a_batch=params.attributions,
            explain_func=self.explain_func
        )

        return results