# XAI_metrics/metrics/faithfulness/sensitivity_n.py
from XAI_metrics.runtime import import_optional_dependency
import numpy as np
from typing import Mapping, Any
from XAI_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

@register_metric
class SensitivityN(BaseMetric):
    NAME = 'SensitivityN'

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

        ctx.model.eval()

        results = quantus.SensitivityN(
            abs=abs_,
            normalise=normalise,
            n_max_percentage=1.0,
            perturb_baseline='uniform' #Con black no hay cambios
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=ctx.attributions
        )

        return results