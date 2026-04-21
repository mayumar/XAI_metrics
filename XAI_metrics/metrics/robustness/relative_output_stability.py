# XAI_metrics/metrics/robustness/relative_output_stability.py
import quantus
from XAI_metrics.base import BaseMetric, MetricContext, register_metric
from typing import Callable, Any, Mapping
import torch.nn as nn
import numpy as np

type ExplainFunc = Callable[[nn.Module, Any, Any | None], np.ndarray]

@register_metric
class RelativeOutputStability(BaseMetric):
    NAME = 'RelativeOutputStability'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
        explain_func: ExplainFunc | None = None
    ):
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("RelativeOutputStability requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func
    
    def run(self):
        ctx = self.context
        p = self.params

        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", False))
        nr_samples = int(p.get("nr_samples", 20))

        ctx.model.eval()

        results = quantus.RelativeOutputStability(
            abs=abs_,
            normalise=normalise,
            nr_samples=nr_samples
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].values,
            y_batch=ctx.y_test.loc[ctx.observations].values,
            a_batch=ctx.attributions,
            explain_func=self.explain_func
        )

        return results