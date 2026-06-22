# xai_metrics/metrics/fidelity/soundness/non_sensitivity.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class NonSensitivity(BaseMetric):
    """
    Quantus Non-Sensitivity metric.

    This metric evaluates whether near-zero attribution values are assigned to
    features on which the model output is not functionally dependent. For each
    observation, Quantus compares features whose attribution is below a
    threshold with features whose perturbation produces only a negligible change
    in the selected model output.

    The score measures disagreement between both sets of features. Lower values
    indicate better agreement between near-zero attributions and features that
    do not significantly affect the model output.

    The metric is based on the Non-Sensitivity metric discussed by Nguyen and
    Rodríguez Martínez (2020), Ancona et al. (2019), and Montavon et al.
    (2018), as implemented in Quantus.
    """
    NAME = 'NonSensitivity'
    
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

            - ``eps`` : float, optional
              Threshold used to decide whether an attribution value is considered
              negligible and whether a prediction change is considered
              insignificant. The default value is ``1e-5``.

            - ``features_in_step`` : int, optional
              Number of features perturbed at each step. The default value is
              ``1``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``True``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            - ``perturb_baseline`` : str, optional
              Baseline value used when perturbing features. Supported values depend
              on the Quantus perturbation function. Common values are ``"black"``,
              ``"white"``, ``"mean"``, ``"random"`` and ``"uniform"``. The default
              value is ``"black"``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default functions provided by Quantus. The default
        perturbation function replaces selected features by the configured
        baseline, and the default aggregation function in Quantus is
        ``numpy.mean``.

        Quantus uses ``eps`` both to identify near-zero attributions and to
        decide whether the model output change after perturbation is
        negligible.
        """
        super().__init__(context, params)


    def run(self):
        """
        Compute the Non-Sensitivity metric.

        The method selects the observations defined in the metric context and
        passes their input data, labels and attribution values to
        :class:`quantus.NonSensitivity`. Quantus perturbs groups of features,
        measures the resulting change in the selected model output and compares
        perturbation-insensitive features with features whose attribution values
        are below ``eps``.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to evaluation mode before the metric is computed.

        Returns
        -------
        List[float]
            Non-Sensitivity score for each evaluated observation. Lower values
            indicate better agreement between near-zero attributions and
            features that do not significantly affect the model output.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        eps = float(p.get("eps", 1e-5))
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

        results = quantus.NonSensitivity(
            eps=eps,
            features_in_step=features_in_step,
            abs=abs_,
            normalise=normalise,
            perturb_baseline=perturb_baseline,
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions
        )

        return results