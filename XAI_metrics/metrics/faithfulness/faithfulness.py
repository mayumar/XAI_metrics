# XAI_metrics/metrics/faithfulness/faithfulness.py
import numpy as np
from aix360.metrics import faithfulness_metric

from XAI_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Any, Mapping, Callable, Dict


@register_metric
class Faithfulness(BaseMetric):
    NAME = "Faithfulness"

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
        base_func: Callable[[Any], Any] | None = None,
        base_func_kwargs: Dict[str, Any] | None = None
    ):
        super().__init__(context, params)
        self.base_func = base_func
        self.base_func_kwargs = base_func_kwargs

    def run(self):
        ctx = self.context
        p = self.params

        model = ctx.model
        X_selected = ctx.X_test.loc[ctx.observations]
        base = self._resolve_base(
            X_reference=ctx.X_test,
            base_values=p.get("base_values"),
            base_strategy=p.get("base_strategy", "mean"),
            base_func=self.base_func,
            base_func_kwargs=self.base_func_kwargs
        )

        scores = []
        for x_row, coefs in zip(X_selected.values, ctx.attributions):
            score = faithfulness_metric(
                model=model,
                x=np.asarray(x_row, dtype=float),
                coefs=np.asarray(coefs, dtype=float),
                base=base,
            )
            scores.append(float(score))

        return scores

    @staticmethod
    def _resolve_base(
        X_reference,
        base_values=None,
        base_strategy="mean",
        base_func: Callable[[Any], Any] | None = None,
        base_func_kwargs: Dict[str, Any] | None = None,
    ):
        if base_values is not None:
            return np.asarray(base_values, dtype=float)

        if base_func is not None:
            kwargs = base_func_kwargs or {}
            return np.asarray(base_func(X_reference, **kwargs), dtype=float)

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