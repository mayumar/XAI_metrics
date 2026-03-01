# XAI_metrics/metrics/fidelity/soundness/non_sensitivity.py
import quantus
from typing import Mapping, Any
from XAI_metrics.base import BaseMetric, MetricContext, register_metric

@register_metric
class NonSensitivity(BaseMetric):
    NAME = 'NonSensitivity'
    
    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        super().__init__(context, params)

    def run(self):
        ctx = self.context
        p = self.params

        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", False))

        ctx.model.eval()

        results = quantus.NonSensitivity(
            abs=abs_,
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].values,
            y_batch=ctx.y_test.loc[ctx.observations].values,
            a_batch=ctx.attributions
        )

        return results