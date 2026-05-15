# XAI_metrics/metrics/faithfulness/faithfulness_estimate.py
import quantus
import numpy as np
from typing import Mapping, Any
from XAI_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped


@register_metric
class FaithfulnessEstimate(BaseMetric):
    NAME = 'FaithfulnessEstimate'

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        super().__init__(context, params)

    def _safe_pearson(self, a, b, batched=False, **kwargs):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if batched:
            scores = []
            for ai, bi in zip(a, b):
                if np.std(ai) == 0 or np.std(bi) == 0:
                    scores.append(0.0)
                else:
                    scores.append(np.corrcoef(ai, bi)[0, 1])
            return np.asarray(scores)

        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        return np.corrcoef(a, b)[0, 1]

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

        results = quantus.FaithfulnessEstimate(
            abs=abs_,
            normalise=normalise,
            perturb_baseline='uniform', # Black y white hace que no se haga ninguna moficiación
            similarity_func=self._safe_pearson # para evitar nan
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=ctx.attributions
        )

        return results