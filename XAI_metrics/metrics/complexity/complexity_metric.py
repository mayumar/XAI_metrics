# XAI_metrics/metrics/complexity/complexity_metric.py
import quantus
from typing import Mapping, Any
from XAI_metrics.base import BaseMetric, MetricContext, register_metric

@register_metric
class Complexity(BaseMetric):
    NAME = 'Complexity'

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        super().__init__(context, params)

    def run(self):
        ctx = self.context
        p = self.params

        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", False))
        
        ctx.model.train()

        results = quantus.Complexity(
            abs=abs_,
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations],
            y_batch=ctx.y_test.loc[ctx.observations],
            a_batch=ctx.attributions
        )

        return results