# xai_metrics/metrics/fidelity/completeness/average_drop.py
from xplique.metrics import fidelity
import numpy as np
from typing import Any, Mapping

from xai_metrics.base import BaseMetric, register_metric, MetricContext


@register_metric
class AverageDropMetric(BaseMetric):
    NAME = "AverageDropMetric"

    def __init__(self, context: MetricContext, params: Mapping[str, Any] | None = None):
        super().__init__(context, params)

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

        metric = fidelity.AverageDropMetric(
            model=ctx.model,
            inputs=X,
            targets=targets,
            batch_size=p.get("batch_size", 64),
            operator=p.get("operator"),
            activation=p.get("activation"),
        )

        return metric.evaluate(attributions)
