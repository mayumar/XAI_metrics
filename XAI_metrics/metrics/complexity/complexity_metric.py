# XAI_metrics/metrics/complexity/complexity_metric.py
from XAI_metrics.runtime import import_optional_dependency
import numpy as np
from typing import Mapping, Any
from XAI_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

@register_metric
class Complexity(BaseMetric):
    NAME = 'Complexity'

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        super().__init__(context, params)

    def run(self):
        ctx = self.context
        p = self.params
        quantus = import_optional_dependency("quantus")

        if np.all(ctx.attributions < 0.0):
            raise MetricSkipped(
                f"{self.NAME} omitida: todas las atribuciones son negativas."
            )

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

        # Normalización
        n_features = ctx.attributions.shape[1]
        max_entropy = np.log(n_features)
        results = np.array(results) / max_entropy

        return results