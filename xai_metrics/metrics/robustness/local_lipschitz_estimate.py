# xai_metrics/metrics/robustness/local_lipschitz_estimate.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Any, Mapping
from xai_metrics.base.types import ExplainFunc

@register_metric
class LocalLipschitzEstimate(BaseMetric):
    """
    Quantus Local Lipschitz Estimate metric.

    This metric evaluates the local stability of an explanation by comparing
    changes in the explanation with changes in the corresponding input. For
    each observation, Quantus generates neighbouring inputs using Gaussian
    noise, recomputes their explanations and calculates the ratio between the
    explanation distance and the input distance. The score is the maximum ratio
    obtained across the sampled neighbours.

    Lower scores indicate that nearby inputs receive similar explanations and
    therefore correspond to more locally robust explanations.

    The metric is based on the Local Lipschitz Estimate proposed by
    Alvarez-Melis and Jaakkola (2018) and implemented in Quantus.
    """
    NAME = 'LocalLipschitzEstimate'

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
            function must accept the model, a batch of inputs and optionally a
            batch of labels or targets, and must return a NumPy array containing
            the generated attributions.
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
              metric. The default value is ``True``.
            - ``perturb_mean`` : float, optional
              Mean of the Gaussian noise used for perturbations. The default value
              is ``0.0``.
            - ``perturb_std`` : float, optional
              Standard deviation of the Gaussian noise used for perturbations. The
              default value is ``0.1``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses the default functions provided by Quantus: the
        Lipschitz constant as the comparison function, Euclidean distance for
        the numerator and denominator, the default attribution normalisation
        function and Gaussian-noise perturbations.

        The score is a finite-sample estimate based on the generated
        neighbours, rather than the exact maximum over the complete local
        neighbourhood.

        Raises
        ------
        ValueError
            If ``explain_func`` is not provided.
        """
        super().__init__(context, params)

        if not explain_func:
            raise ValueError("LocalLipschitzEstimate requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func

    
    def run(self):
        """
        Compute the Local Lipschitz Estimate metric.

        The method passes the selected inputs, target labels, original
        attributions and explanation function to
        :class:`quantus.LocalLipschitzEstimate`. Quantus samples neighbouring
        inputs, recomputes their explanations and returns the maximum ratio
        between explanation and input changes for each observation.

        If all attribution values are negative, their absolute values are used
        when ``abs=True``; otherwise, the metric is skipped. The model is set
        to training mode and the device stored in the context is forwarded to
        Quantus.

        Returns
        -------
        List[float]
            Local Lipschitz Estimate score for each evaluated observation.
            Lower values indicate greater local explanation stability.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        nr_samples = int(p.get("nr_samples", 200))
        abs_ = bool(p.get("abs", False))
        normalise = bool(p.get("normalise", True))
        perturb_mean = float(p.get("perturb_mean", 0.0))
        perturb_std = float(p.get("perturb_std", 0.1))

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.train()

        results = quantus.LocalLipschitzEstimate(
            nr_samples=nr_samples,
            abs=abs_,
            normalise=normalise,
            perturb_mean=perturb_mean,
            perturb_std=perturb_std
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions,
            explain_func=self.explain_func,
            device=ctx.device
        )

        return results