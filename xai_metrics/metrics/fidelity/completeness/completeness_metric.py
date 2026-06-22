# xai_metrics/metrics/fidelity/completeness/completeness_metric.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class Completeness(BaseMetric):
    """
    Quantus Completeness metric.

    This metric evaluates whether the sum of the attribution values matches the
    difference between the model output for the original input and the model
    output for a baseline input. This property is also known as Summation to
    Delta or Conservation.

    The metric returns one boolean value per observation. A value of ``True``
    indicates that the attribution sum satisfies the completeness condition for
    the selected model output.

    Higher scores are better when the boolean results are aggregated, since a
    larger proportion of ``True`` values indicates stronger agreement with the
    completeness axiom.

    The metric is based on the Completeness property proposed by Sundararajan
    et al. (2017), the Summation to Delta property proposed by Shrikumar et al.
    (2017), and the Conservation property discussed by Montavon et al. (2018),
    as implemented in Quantus.
    """
    NAME = 'Completeness'

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

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``False``.
            
            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.
            
            - ``perturb_baseline`` : str, optional
              Baseline value used to create the reference input. Supported values
              depend on the Quantus perturbation function. Common values are
              ``"black"``, ``"white"``, ``"mean"``, ``"random"`` and
              ``"uniform"``. The default value is ``"black"``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default functions provided by Quantus. The default
        perturbation function replaces all features by the configured baseline,
        and the default output transformation is the identity function.

        Quantus computes this metric using logits by default, not softmax
        probabilities.
        """
        super().__init__(context, params)


    def run(self):
        """
        Compute the Completeness metric.

        The method selects the observations defined in the metric context and
        passes their input data, labels and attribution values to
        :class:`quantus.Completeness`. Quantus replaces the input features by a
        baseline, computes the difference between the model output at the
        original input and at the baseline input, and compares this difference
        with the sum of the attribution values.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to evaluation mode before the metric is computed.

        Returns
        -------
        List[bool]
            Completeness result for each evaluated observation. ``True``
            indicates that the sum of the attribution values matches the output
            difference between the original input and the baseline input.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        abs_ = bool(p.get("abs", False))
        normalise = bool(p.get("normalise", True))
        perturb_baseline = str(p.get("perturb_baseline", "black"))

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.eval()

        results = quantus.Completeness(
            abs=abs_,
            normalise=normalise,
            perturb_baseline=perturb_baseline
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions
        )

        return results