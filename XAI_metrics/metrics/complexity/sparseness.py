# XAI_metrics/metrics/complexity/sparseness.py
import quantus
import numpy as np
from typing import Mapping, Any
from XAI_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

@register_metric
class Sparseness(BaseMetric):
    NAME = 'Sparseness'

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        super().__init__(context, params)

    def run(self):
        ctx = self.context
        p = self.params

        if np.all(ctx.attributions < 0.0):
            raise MetricSkipped(
                f"{self.NAME} omitida: todas las atribuciones son negativas."
            )

        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", False))
        
        ctx.model.train()

        results = quantus.Sparseness(
            abs=abs_,
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations],
            y_batch=ctx.y_test.loc[ctx.observations],
            a_batch=ctx.attributions
        )

        return results