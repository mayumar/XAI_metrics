# XAI_metrics/metrics/robustness/local_lipschitz_estimate.py
import quantus
import numpy as np
import torch.nn as nn

from XAI_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Callable, Any, Mapping, Dict
type ExplainFunc = Callable[[nn.Module, Any, Any | None], np.ndarray]

@register_metric
class LocalLipschitzEstimate(BaseMetric):
    NAME = 'LocalLipschitzEstimate'

    def __init__(
        self,
        context: MetricContext,
        similarity_func: Callable[[Any], Any],
        params: Mapping[str, Any] | None = None,
        explain_func: ExplainFunc | None = None,
        norm_numerator: Callable[[Any], Any] | None = None,
        norm_denominator: Callable[[Any], Any] | None = None,
        normalise_func: Callable[..., np.ndarray] | None = None,
        normalise_func_kwargs: Dict[str, Any] | None = None,
        perturb_func: Callable[[Any], Any] | None = None,
        perturb_func_kwargs: Dict[str, Any] | None = None
    ):
        super().__init__(context, params)

        if not explain_func:
            raise ValueError("LocalLipschitzEstimate requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func

        self.similarity_func = similarity_func
        self.norm_numerator = norm_numerator
        self.norm_denominator = norm_denominator
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = perturb_func_kwargs
    
    def run(self):
        ctx = self.context
        p = self.params

        if np.all(ctx.attributions < 0.0):
            raise MetricSkipped(
                f"{self.NAME} skipped: all attributions are negative."
            )

        nr_samples = int(p.get("nr_samples", 200))
        abs_ = bool(p.get("abs", False))
        normalise = bool(p.get("normalise", True))
        perturb_mean = float(p.get("perturb_mean", 0.0))
        perturb_std = float(p.get("perturb_std", 0.1))

        ctx.model.train()

        results = quantus.LocalLipschitzEstimate(
            similarity_func=self.similarity_func,
            norm_numerator=self.norm_numerator,
            abs=abs_,
            normalise=normalise,
            nr_samples=nr_samples
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=ctx.attributions,
            explain_func=self.explain_func
        )

        return results