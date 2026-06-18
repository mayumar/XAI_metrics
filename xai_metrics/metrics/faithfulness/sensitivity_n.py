# xai_metrics/metrics/faithfulness/sensitivity_n.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class SensitivityN(BaseMetric):
    """
    Quantus Sensitivity-N metric.

    This metric evaluates the agreement between feature attribution values and
    the variation in the target model output caused by perturbing the
    corresponding features.

    Quantus orders features by decreasing attribution value and progressively
    perturbs them in groups. At each step, it compares the target-output change
    with the attribution sum of the processed feature group using Pearson
    correlation across the evaluated observations.

    The number of perturbation steps is limited by ``n_max_percentage``.
    Higher correlation values indicate stronger agreement between the
    explanation and the model behaviour.

    The metric is based on the Sensitivity-N test proposed by Ancona et al.
    (2018) and implemented in Quantus.
    """
    NAME = 'SensitivityN'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model, ``X_test``,
            ``y_test``, selected observations and attribution values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``n_max_percentage`` : float, optional
              Maximum percentage of features to evaluate. The default value is
              ``0.8``.

            - ``features_in_step`` : int, optional
              Number of features perturbed at each step. The default value is
              ``1``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is
              ``False``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            - ``perturb_baseline`` : str, optional
              Baseline value used when perturbing features. Supported values
              depend on the Quantus perturbation function. Common values are
              ``"black"``, ``"white"``, ``"mean"``, ``"random"`` and
              ``"uniform"``. The default value is ``"black"``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default Pearson correlation, normalisation and
        baseline-replacement functions provided by Quantus. Quantus aggregates
        the step-wise correlations by default.
        """
        super().__init__(context, params)


    def run(self):
        """
        Compute the Sensitivity-N metric.

        The method passes the selected input data, target labels and attribution
        values to :class:`quantus.SensitivityN`. Quantus progressively perturbs
        feature groups ordered by decreasing attribution value and computes the
        Pearson correlation between their attribution sums and the resulting
        target-output changes.

        Only the perturbation steps covered by ``n_max_percentage`` are
        evaluated. If all attribution values are negative, their absolute
        values are used when ``abs=True``; otherwise, the metric is skipped.
        The model is set to evaluation mode before the computation.

        Returns
        -------
        List[float]
            Sensitivity-N result returned by Quantus. Higher values indicate
            stronger agreement between attribution values and target-output
            changes. With the default Quantus configuration, the step-wise
            correlations are aggregated.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        n_max_percentage = float(p.get("n_max_percentage", 0.8))
        features_in_step = int(p.get("features_in_step", 1))
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

        results = quantus.SensitivityN(
            n_max_percentage=n_max_percentage,
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
