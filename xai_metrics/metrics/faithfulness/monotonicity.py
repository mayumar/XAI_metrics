# xai_metrics/metrics/faithfulness/monotonicity.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class Monotonicity(BaseMetric):
    """
    Quantus Monotonicity metric.

    This metric evaluates whether the target model output increases
    monotonically as features are progressively introduced from a baseline
    input. Features are processed in increasing order of attribution value and
    may be introduced individually or in groups.

    For each observation, the metric returns ``True`` when the sequence of
    target outputs is monotonically non-decreasing and ``False`` otherwise.

    The metric is based on the Monotonicity metric proposed by Arya et al.
    (2019) and the monotonic attribute functions described by Luss et al.
    (2019), as implemented in Quantus.
    """
    NAME = 'Monotonicity'

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

            - ``features_in_step`` : int, optional
              Number of features added at each perturbation step. The default
              value is ``1``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``True``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            - ``perturb_baseline`` : str, optional
              Baseline value used to initialise the perturbed input. Supported
              values depend on the Quantus perturbation function. Common values
              are ``"black"``, ``"white"``, ``"mean"``, ``"random"`` and
              ``"uniform"``. The default value is ``"black"``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default normalisation and perturbation functions
        provided by Quantus.
        """
        super().__init__(context, params)


    def run(self):
        """
        Compute the Monotonicity metric.

        The method passes the selected input data, target labels and attribution
        values to :class:`quantus.Monotonicity`. For each observation, Quantus
        starts from a baseline input, processes features in increasing order of
        attribution value and evaluates the target model output after each
        group is introduced.

        If all attribution values are negative, their absolute values are used
        when ``abs=True``; otherwise, the metric is skipped. The model is set to
        evaluation mode before the computation.

        Returns
        -------
        List[bool]
            Monotonicity result for each evaluated observation. ``True``
            indicates that the target output is monotonically non-decreasing
            across the feature-introduction steps.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        features_in_step = int(p.get("features_in_step", 1))
        abs_ = bool(p.get("abs", True))
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

        results = quantus.Monotonicity(
            features_in_step=features_in_step,
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
