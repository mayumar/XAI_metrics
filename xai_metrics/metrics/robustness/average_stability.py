# xai_metrics/metrics/robustness/average_stability.py
from xplique.metrics import AverageStability as XpliqueAverageStability
import numpy as np
from typing import Any, Mapping

from xai_metrics.base import BaseMetric, register_metric, MetricContext
from xai_metrics.base.types import ExplainFunc


@register_metric
class AverageStability(BaseMetric):
    NAME = "AverageStability"

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None,
    ):
        super().__init__(context, params)
        if explain_func is None:
            raise ValueError("AverageStability requires 'explain_func' to be provided via dependencies.")
        self.explain_func = explain_func

    def run(self):
        ctx = self.context
        p = self.params

        X = ctx.X_test.loc[ctx.observations].to_numpy(dtype=np.float32, copy=True)
        y = np.asarray(ctx.y_test.loc[ctx.observations]).astype(int).ravel()
        attributions = np.asarray(ctx.attributions, dtype=np.float32)

        if bool(p.get("one_hot_targets", False)):
            num_classes = int(p.get("num_classes", np.max(y) + 1))
            targets = np.zeros((len(y), num_classes), dtype=np.float32)
            targets[np.arange(len(y)), y] = 1.0
        else:
            targets = y

        metric = XpliqueAverageStability(
            model=ctx.model,
            inputs=X,
            targets=targets,
            batch_size=p.get("batch_size", 64),
            radius=float(p.get("radius", p.get("noise_std", 0.1))),
            distance=p.get("distance", "l2"),
            nb_samples=int(p.get("nb_samples", p.get("n_perturbations", 20))),
        )

        return metric.evaluate(
            self.explain_func,
            base_explanations=attributions,
        )
