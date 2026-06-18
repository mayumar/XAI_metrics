# xai_metrics/metrics/faithfulness/consistency.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class Consistency(BaseMetric):
    """
    Quantus Consistency metric.

    This metric evaluates whether observations with equivalent explanations
    exhibit consistent model behaviour. Continuous attribution vectors are
    discretised by Quantus and observations sharing the same discrete
    representation are grouped together. For each observation, the metric
    computes the proportion of other observations in its group that receive
    the same predicted class.

    Higher scores indicate stronger agreement between similar explanations and
    model predictions.

    The metric is based on the Consistency metric proposed by Dasgupta et al.
    (2022) and implemented in Quantus.
    """
    NAME = 'Consistency'

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

            - ``abs`` : bool, optional
              Whether to use absolute attribution values before computing the
              metric. If all attribution values are negative and this parameter
              is ``True``, their absolute values are used. If it is ``False``,
              the metric is skipped. The default value is ``True``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before discretising
              the explanations. The default value is ``True``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default discretisation and normalisation functions
        provided by Quantus. The default discretisation function is
        ``top_n_sign``.
        """
        super().__init__(context, params)
    
    
    def run(self):
        """
        Compute the Consistency metric.

        The method selects the observations defined in the metric context and
        passes their input data, labels and attribution values to
        :class:`quantus.Consistency`. Quantus discretises the explanations,
        predicts the class of each observation and compares predictions among
        observations with equivalent discrete explanations.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to evaluation mode before the metric is computed.

        Returns
        -------
        List[float]
            Consistency score for each evaluated observation. Higher values
            indicate that observations with the same discretised explanation
            more frequently receive the same predicted class.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

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

        results = quantus.Consistency(
            abs=abs_,
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions
        )

        return results
