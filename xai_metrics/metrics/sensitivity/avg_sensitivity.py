# XAI_metrics/metrics/sensitivity/avg_sensitivity.py
import quantus
import numpy as np
from quantus.functions.perturb_func import batch_uniform_noise

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Callable, Any, Mapping, Dict
from xai_metrics.base.types import ExplainFunc

@register_metric
class AvgSensitivity(BaseMetric):
    """
    Quantus Average Sensitivity metric.

    This metric evaluates explanation robustness by repeatedly perturbing each
    input, recomputing its explanation with ``explain_func``, and measuring the
    average explanation change across the sampled perturbations.

    The metric is based on the Average Sensitivity metric proposed by Yeh et al.
    (2019) and also discussed by Bhatt et al. (2020), as implemented in
    Quantus.
    """
    NAME = 'AvgSensitivity'

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
        perturb_func_kwargs: Dict[str, Any] | None = None,
    ):
        """
        Initialize the Average Sensitivity metric.

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
        similarity_func : Callable[..., numpy.ndarray] or None, optional
            Function used to compare the original and perturbed attribution values.
            The function must be compatible with the Quantus similarity interface.
            If ``None``, Quantus uses its default difference function.
        norm_numerator : Callable[..., numpy.ndarray] or None, optional
            Function used to compute the norm of the explanation difference in the
            numerator of the sensitivity ratio. If ``None``, Quantus uses its
            default Frobenius norm.
        norm_denominator : Callable[..., numpy.ndarray] or None, optional
            Function used to compute the norm of the original attribution values in
            the denominator of the sensitivity ratio. If ``None``, Quantus uses its
            default Frobenius norm.
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
            ``None``, ``quantus.functions.perturb_func.batch_uniform_noise`` is
            used.
        perturb_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``perturb_func``. If ``None``, no
            additional keyword arguments are passed.

        Raises
        ------
        ValueError
            If ``explain_func`` is not provided.
        """
        super().__init__(context, params)

        if explain_func is None:
            raise ValueError("AvgSensitivity requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func

        self.similarity_func = similarity_func
        self.norm_numerator = norm_numerator
        self.norm_denominator = norm_denominator
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs
        self.perturb_func = perturb_func or batch_uniform_noise
        self.perturb_func_kwargs = perturb_func_kwargs
    
    def run(self):
        """
        Compute the Average Sensitivity metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes them
        to :class:`quantus.AvgSensitivity`. The model is set to training mode before
        computing the metric, and ``ctx.device`` is forwarded to Quantus when
        available.

        Returns
        -------
        List[float]
            Average Sensitivity score for each evaluated observation. Lower values
            indicate more robust explanations under random input perturbations.

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
        normalise = bool(p.get("normalise", False))
        lower_bound = float(p.get("lower_bound", 0.2))
        upper_bound = p.get("upper_bound")
        if upper_bound is not None:
            upper_bound = float(upper_bound)

        ctx.model.train()

        results = quantus.AvgSensitivity(
            similarity_func=self.similarity_func,
            norm_numerator=self.norm_numerator,
            norm_denominator=self.norm_denominator,
            nr_samples=nr_samples,
            abs=abs_,
            normalise=normalise,
            normalise_func=self.normalise_func,
            normalise_func_kwargs=self.normalise_func_kwargs,
            perturb_func=self.perturb_func,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
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