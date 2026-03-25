from typing import Any, Mapping

import numpy as np

from XAI_metrics.base import BaseMetric, MetricContext, register_metric


@register_metric
class MonotonicityMetric(BaseMetric):
    NAME = "MonotonicityMetric"

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
    ):
        super().__init__(context, params)

    def run(self):
        from aix360.metrics import monotonicity_metric

        ctx = self.context
        p = self.params

        model = ctx.extras.get("prediction_model", ctx.model)
        X_selected = ctx.X_test.loc[ctx.observations]
        base = self._resolve_base(
            X_reference=ctx.extras.get("X_reference", ctx.X_test),
            base_values=p.get("base_values"),
            base_strategy=p.get("base_strategy", "mean"),
        )

        scores = []
        for x_row, coefs in zip(X_selected.values, ctx.attributions):
            score = monotonicity_metric(
                model=model,
                x=np.asarray(x_row, dtype=float),
                coefs=np.asarray(coefs, dtype=float),
                base=base,
            )
            scores.append(bool(score))

        return scores

    @staticmethod
    def _resolve_base(X_reference, base_values=None, base_strategy="mean"):
        if base_values is not None:
            return np.asarray(base_values, dtype=float)

        values = (
            X_reference.values
            if hasattr(X_reference, "values")
            else np.asarray(X_reference, dtype=float)
        )

        if base_strategy == "mean":
            return np.mean(values, axis=0)
        if base_strategy == "median":
            return np.median(values, axis=0)
        if base_strategy == "zero":
            return np.zeros(values.shape[1], dtype=float)

        raise ValueError(f"Unknown base_strategy: {base_strategy}")