# xai_metrics/metrics/faithfulness/sufficiency.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class Sufficiency(BaseMetric):
    """
    Quantus Sufficiency metric.

    This metric evaluates whether observations with similar explanations
    receive the same model prediction. Quantus computes the pairwise distances
    between attribution vectors and considers two explanations similar when
    their distance is less than or equal to ``threshold``.

    For each observation, the score is the proportion of other observations
    with similar explanations that receive the same predicted class. A score
    of ``0.0`` is returned when no other explanation satisfies the distance
    threshold.

    Higher scores indicate stronger agreement between similar explanations and
    model predictions. Since explanations are compared within the evaluated
    batch, the results depend on the selected observations.

    The metric is based on the Sufficiency test proposed by Dasgupta et al.
    (2022) and implemented in Quantus.
    """
    NAME = 'Sufficiency'

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
            ``X_test``, ``y_test``, selected observations and attribution values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``threshold`` : float, optional
              Maximum distance between two attribution vectors for their
              explanations to be considered similar. The default value is
              ``0.6``.

            - ``distance_func`` : str, optional
              Distance function used to compare attribution vectors. The value
              is passed to Quantus and is typically a valid SciPy distance name,
              such as ``"seuclidean"``. The default value is ``"seuclidean"``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is
              ``True``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default normalisation function provided by
        Quantus. Scores depend on the observations evaluated in the same call,
        since these observations define the explanation neighbourhoods.
        """
        super().__init__(context, params)
        

    def run(self):
        """
        Compute the Sufficiency metric.

        The method passes the selected input data, labels and attribution values
        to :class:`quantus.Sufficiency`. Quantus compares the attribution
        vectors, identifies observations with similar explanations and computes
        the proportion that receive the same predicted class. Self-comparisons
        are excluded.

        If all attribution values are negative, their absolute values are used
        when ``abs=True``; otherwise, the metric is skipped. The model is set to
        evaluation mode before the computation.

        Returns
        -------
        List[float]
            Sufficiency score for each evaluated observation. Scores range from
            ``0.0`` to ``1.0``. Higher values indicate that observations with
            similar explanations more frequently receive the same predicted
            class.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        threshold = float(p.get("threshold", 0.6))
        distance_func = str(p.get("distance_func", "seuclidean"))
        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", True))

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.eval()

        results = quantus.Sufficiency(
            threshold=threshold,
            distance_func=distance_func,
            abs=abs_,
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions
        )

        return results
