# xai_metrics/metrics/complexity/sparseness.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Mapping, Any

@register_metric
class Sparseness(BaseMetric):
    """
    Quantus Sparseness metric.

    This metric measures how concentrated the attribution magnitude is across
    the input features using the Gini index. Quantus applies the absolute-value
    operation to the attributions before computing the score.

    Higher scores indicate that importance is concentrated on a smaller subset
    of features and therefore correspond to sparser explanations. Lower scores
    indicate a more uniform distribution of attribution magnitude.

    The metric is based on the Sparseness metric proposed by Chalasani et al.
    (2020) and implemented in Quantus.
    """
    NAME = 'Sparseness'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, ``y_test``, selected observations and attribution
            values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default normalisation function provided by
        Quantus. Quantus applies the absolute-value operation to the
        attributions because its ``abs`` parameter is not exposed by this
        wrapper and defaults to ``True``.
        """
        super().__init__(context, params)

    def run(self):
        """
        Compute the Sparseness metric.

        The method passes the selected input data, labels and attribution
        values to :class:`quantus.Sparseness`. Quantus flattens each
        explanation, takes the absolute attribution values, sorts them and
        computes their Gini index.

        If all attribution values are negative, this wrapper converts them to
        absolute values before calling Quantus. The model is set to training
        mode following the current wrapper implementation, although the metric
        itself depends only on the attribution values.

        Returns
        -------
        List[float]
            Sparseness score for each evaluated observation. Higher values
            indicate that attribution magnitude is concentrated on fewer
            features, while lower values indicate a more uniform distribution.
        """
        ctx = self.context
        p = self.params

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            attributions = np.abs(attributions)

        normalise = bool(p.get("normalise", True))
        
        ctx.model.train()

        results = quantus.Sparseness(
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations],
            y_batch=ctx.y_test.loc[ctx.observations],
            a_batch=attributions
        )

        return results