# XAI_metrics/metrics/robustness/local_lipschitz_estimate.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Callable, Any, Mapping, Dict
from xai_metrics.base.types import ExplainFunc

@register_metric
class LocalLipschitzEstimate(BaseMetric):
    """
    Quantus Local Lipschitz Estimate metric.

    This metric evaluates explanation stability in a local neighbourhood of
    each input. For each observation, Quantus samples perturbed versions of the
    input, recomputes explanations with ``explain_func``, and estimates the
    maximum ratio between explanation changes and input changes.

    The metric is based on the Local Lipschitz Estimate, also referred to as a
    stability test, proposed by Alvarez-Melis and Jaakkola (2018), as
    implemented in Quantus.
    """
    NAME = 'LocalLipschitzEstimate'

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None,
        similarity_func: Callable[..., np.ndarray] | None = None,
        norm_numerator: Callable[..., np.ndarray] | None = None,
        norm_denominator: Callable[..., np.ndarray] | None = None,
        normalise_func: Callable[..., np.ndarray] | None = None,
        normalise_func_kwargs: Dict[str, Any] | None = None,
        perturb_func: Callable[..., np.ndarray] | None = None,
        perturb_func_kwargs: Dict[str, Any] | None = None
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
        similarity_func : Callable[..., numpy.ndarray] or None, optional
            Function used to compute the local Lipschitz estimate from explanation
            changes and input changes. The function must be compatible with the
            Quantus similarity interface, accepting arguments such as ``a``, ``b``,
            ``c``, ``d``, ``norm_numerator`` and ``norm_denominator``. If
            ``None``, Quantus uses its default Lipschitz constant function.
        norm_numerator : Callable[..., numpy.ndarray] or None, optional
            Function used to compute the norm of the explanation difference in the
            numerator of the Lipschitz ratio. If ``None``, Quantus uses its default
            Euclidean distance.
        norm_denominator : Callable[..., numpy.ndarray] or None, optional
            Function used to compute the norm of the input difference in the
            denominator of the Lipschitz ratio. If ``None``, Quantus uses its
            default Euclidean distance.
        normalise_func : Callable[..., numpy.ndarray] or None, optional
            Custom normalisation function passed to Quantus. The function must
            accept the attribution array as its first argument and may accept
            additional keyword arguments from ``normalise_func_kwargs``. If
            ``None``, Quantus uses its default normalisation behaviour when
            ``normalise=True``.
        normalise_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``normalise_func`` when normalisation is
            enabled. If ``None``, no additional keyword arguments are passed.
        perturb_func : Callable[..., numpy.ndarray] or None, optional
            Perturbation function passed to Quantus. The function must be
            compatible with Quantus perturbation functions, accepting at least an
            input array and feature indices, and returning the perturbed array. If
            ``None``, Quantus uses its default Gaussian-noise perturbation
            function.
        perturb_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``perturb_func``. If ``None``, no
            additional keyword arguments are passed.

        Raises
        ------
        ValueError
            If ``explain_func`` is not provided.
        """
        super().__init__(context, params)

        if not explain_func:
            raise ValueError("LocalLipschitzEstimate requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func

        self.similarity_func = similarity_func
        self.norm_numerator = norm_numerator
        self.norm_denominator = norm_denominator
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = perturb_func_kwargs
    
    def run(self):
        """
        Compute the Local Lipschitz Estimate metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes them
        to :class:`quantus.LocalLipschitzEstimate`. The model is set to training
        mode before computing the metric, and ``ctx.device`` is forwarded to
        Quantus when available.

        Returns
        -------
        List[float]
            Local Lipschitz Estimate score for each evaluated observation. Lower
            values indicate more stable explanations under local input
            perturbations.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative, since the metric is skipped for
            that attribution configuration.
        """
        ctx = self.context
        p = self.params

        if np.all(ctx.attributions < 0.0):
            raise MetricSkipped(
                f"{self.NAME} skipped: all attributions are negative."
            )

        nr_samples = int(p.get("nr_samples", 200))
        abs_ = bool(p.get("abs", False))
        normalise = bool(p.get("normalise", True))
        perturb_mean = float(p.get("perturb_mean", 0.0))
        perturb_std = float(p.get("perturb_std", 0.1))

        ctx.model.train()

        results = quantus.LocalLipschitzEstimate(
            similarity_func=self.similarity_func,
            norm_numerator=self.norm_numerator,
            norm_denominator=self.norm_denominator,
            nr_samples=nr_samples,
            abs=abs_,
            normalise=normalise,
            normalise_func=self.normalise_func,
            normalise_func_kwargs=self.normalise_func_kwargs,
            perturb_func=self.perturb_func,
            perturb_mean=perturb_mean,
            perturb_std=perturb_std,
            perturb_func_kwargs=self.perturb_func_kwargs
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=ctx.attributions,
            explain_func=self.explain_func,
            device=ctx.device
        )

        return results