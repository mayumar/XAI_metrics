# xai_metrics/metrics/complexity/complexity_metric.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Mapping, Any

@register_metric
class Complexity(BaseMetric):
    """
    Quantus Complexity metric.

    This metric measures explanation complexity using the entropy of the
    attribution distribution. Quantus takes the absolute attribution values
    and expresses each feature contribution as a fraction of the total
    attribution magnitude.

    Higher scores indicate that importance is distributed across more features
    and therefore corresponds to more complex explanations. Lower scores
    indicate that importance is concentrated on fewer features.

    For an explanation with ``n`` features, the maximum entropy is
    approximately ``log(n)`` when importance is distributed uniformly. The
    scores returned by this wrapper are not divided by this maximum value.

    The metric is based on the Complexity metric proposed by Bhatt et al.
    (2020) and implemented in Quantus.
    """
    NAME = 'Complexity'

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
        Quantus. Quantus also applies the absolute-value operation to the
        attributions because its ``abs`` parameter is not exposed by this
        wrapper and defaults to ``True``.
        """
        super().__init__(context, params)

    def run(self):
        """
        Compute the Complexity metric.

        The method passes the selected input data, labels and attribution
        values to :class:`quantus.Complexity`. Quantus flattens each explanation,
        takes its absolute values and computes the entropy of the fractional
        feature contributions.

        If all attribution values are negative, this wrapper converts them to
        absolute values before calling Quantus. The model is set to training
        mode following the current wrapper implementation, although the metric
        itself depends only on the attribution values.

        Returns
        -------
        List[float]
            Complexity score for each evaluated observation. Lower values
            indicate that importance is concentrated on fewer features, while
            higher values indicate that it is more widely distributed. Scores
            are not normalised by the theoretical maximum ``log(n)``.
        """
        ctx = self.context
        p = self.params

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            attributions = np.abs(attributions)

        normalise = bool(p.get("normalise", True))
        
        ctx.model.train()

        results = quantus.Complexity(
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations],
            y_batch=ctx.y_test.loc[ctx.observations],
            a_batch=attributions
        )

        # Normalización
        # n_features = attributions.shape[1]
        # max_entropy = np.log(n_features)
        # results = np.array(results) / max_entropy

        return results