# xai_metrics/metrics/robustness/relative_input_stability.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Any, Mapping
from xai_metrics.base.types import ExplainFunc

@register_metric
class RelativeInputStability(BaseMetric):
    """
    Quantus Relative Input Stability metric.

    This metric evaluates explanation robustness by comparing the relative
    change in an explanation with the relative change in its input. For each
    observation, Quantus generates several perturbed inputs, recomputes their
    explanations and returns the maximum ratio between both relative changes.

    Lower scores indicate more stable explanations, while higher scores
    indicate greater sensitivity to input perturbations.

    The metric is based on Relative Input Stability proposed by Agarwal et al.
    (2022), as implemented in Quantus.
    """
    NAME = 'RelativeInputStability'

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, ``y_test``, selected observations, attribution values and
            optional device information.
        explain_func : ExplainFunc
            Function used to generate explanations for perturbed inputs. The
            function must be compatible with Quantus explanation functions and
            return a NumPy array containing the generated attributions.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``nr_samples`` : int, optional
              Number of perturbed samples generated for each observation. The
              default value is ``200``.
            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``False``.
            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``False``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        Quantus uses uniform-noise perturbations with an upper bound of ``0.2``.
        When normalisation is enabled, the default function is
        ``normalise_by_average_second_moment_estimate``.

        A constant of ``1e-6`` is used to avoid division by zero. By default,
        perturbations that change the model prediction produce ``nan`` scores.

        Raises
        ------
        ValueError
            If ``explain_func`` is not provided.
        """
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("RelativeInputStability requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func
    
    def run(self):
        """
        Compute the Relative Input Stability metric.

        The method passes the selected inputs, target labels, original
        attributions and explanation function to
        :class:`quantus.RelativeInputStability`. Quantus perturbs each input,
        recomputes its explanation and returns the maximum ratio between the
        relative explanation change and the relative input change.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to evaluation mode before the metric is computed. The
        device stored in the metric context is forwarded to Quantus.

        Returns
        -------
        list[float]
            Relative Input Stability score for each evaluated observation.
            Lower values indicate more stable explanations. A score may be
            ``nan`` when a perturbation changes the model prediction.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        nr_samples = int(p.get("nr_samples", 200))
        abs_ = bool(p.get("abs", False))
        normalise = bool(p.get("normalise", False))

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.eval()

        results = quantus.RelativeInputStability(
            nr_samples=nr_samples,
            abs=abs_,
            normalise=normalise
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions,
            explain_func=self.explain_func,
            device=ctx.device
        )

        return results