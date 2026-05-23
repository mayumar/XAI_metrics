# XAI_metrics/metrics/faithfulness/monotonicity_correlation.py
from XAI_metrics.runtime import import_optional_dependency
import numpy as np
from scipy.stats import spearmanr
from typing import Mapping, Any
from XAI_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

@register_metric
class MonotonicityCorrelation(BaseMetric):
    NAME = 'MonotonicityCorrelation'

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        super().__init__(context, params)

    def _safe_spearman(self, a, b, batched=False, **kwargs):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if batched:
            scores = []
            for ai, bi in zip(a, b):
                if np.std(ai) == 0 or np.std(bi) == 0:
                    scores.append(0.0)
                else:
                    scores.append(spearmanr(ai, bi).correlation)
            return np.asarray(scores)

        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        return spearmanr(a, b).correlation

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

        results = quantus.MonotonicityCorrelation(
            abs=abs_,
            normalise=normalise,
            similarity_func=self._safe_spearman
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=ctx.attributions
        )

        return results