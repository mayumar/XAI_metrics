# xai_metrics/metrics/robustness/max_sensitivity.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Any, Mapping
from xai_metrics.base.types import ExplainFunc

@register_metric
class MaxSensitivity(BaseMetric):
    """
    Quantus Max-Sensitivity metric.

    This metric evaluates explanation robustness by measuring the largest
    relative change in an explanation under small input perturbations. For each
    observation, Quantus generates several perturbed inputs, recomputes their
    explanations and compares them with the original explanation. The score is
    the maximum ratio between the norm of the explanation change and the norm
    of the original explanation.

    Lower scores indicate more robust explanations, while higher scores
    indicate greater sensitivity to input perturbations.

    The metric is based on Max-Sensitivity proposed by Yeh et al. (2019) and
    discussed by Bhatt et al. (2020), as implemented in Quantus.
    """
    NAME = 'MaxSensitivity'

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
            - ``lower_bound`` : float, optional
              Lower bound of the uniform noise used for perturbations. The default
              value is ``0.2``.
            - ``upper_bound`` : float or None, optional
              Upper bound of the uniform noise used for perturbations. If ``None``,
              Quantus uses its default behaviour. The default value is ``None``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default functions provided by Quantus:
        element-wise difference to compare explanations, the Frobenius norm for
        the numerator and denominator, the default normalisation function and
        uniform-noise perturbations.

        Raises
        ------
        ValueError
            If ``explain_func`` is not provided.
        """
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("MaxSensitivity requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func
        
    
    def run(self):
        """
        Compute the Max-Sensitivity metric.

        The method passes the selected inputs, target labels, original
        attributions and explanation function to
        :class:`quantus.MaxSensitivity`. Quantus repeatedly perturbs each input,
        recomputes its explanation and returns the maximum relative explanation
        change across the configured perturbations.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to training mode before the metric is computed. The
        device stored in the metric context is forwarded to Quantus.

        Returns
        -------
        List[float]
            Max-Sensitivity score for each evaluated observation. Lower values
            indicate explanations that are more robust to small random input
            perturbations.

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
        lower_bound = float(p.get("lower_bound", 0.2))
        upper_bound = p.get("upper_bound")
        if upper_bound is not None:
            upper_bound = float(upper_bound)

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)


        ctx.model.train()

        results = quantus.MaxSensitivity(
            nr_samples=nr_samples,
            abs=abs_,
            normalise=normalise,
            lower_bound=lower_bound,
            upper_bound=upper_bound
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions,
            explain_func=self.explain_func,
            device=ctx.device
        )

        return results