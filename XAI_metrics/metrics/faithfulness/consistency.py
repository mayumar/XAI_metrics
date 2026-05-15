# XAI_metrics/metrics/faithfulness/consistency.py
import quantus
import numpy as np
from typing import Mapping, Any
from XAI_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

@register_metric
class Consistency(BaseMetric):
    NAME = 'Consistency'

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

        ctx.model.eval()

        results = quantus.Consistency(
            abs=abs_,
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=ctx.attributions
        )

        return results