# XAI_metrics/metrics/robustness/relative_output_stability.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Callable, Any, Mapping, Dict
from xai_metrics.base.types import ExplainFunc

@register_metric
class RelativeOutputStability(BaseMetric):
    """
    Quantus Relative Output Stability metric.

    This metric evaluates the stability of explanations with respect to changes
    in the model output. For each observation, Quantus generates perturbed
    inputs, recomputes their explanations with ``explain_func``, and computes
    the maximum ratio between the relative explanation change and the change in
    model output logits.

    The metric is based on the Relative Output Stability metric proposed by
    Agarwal et al. (2022), as implemented in Quantus.
    """
    NAME = 'RelativeOutputStability'

    def __init__(
        self,
        context: MetricContext,
        explain_func: ExplainFunc,
        params: Mapping[str, Any] | None = None,
        normalise_func: Callable[..., np.ndarray] | None = None,
        normalise_func_kwargs: Dict[str, Any] | None = None,
        perturb_func: Callable[..., np.ndarray] | None = None,
        perturb_func_kwargs: Dict[str, Any] | None = None,
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
            ``None``, Quantus uses its default uniform-noise perturbation function.
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
            raise ValueError("RelativeOutputStability requires 'explain_func' to be provided via dependencies.")

        self.explain_func = explain_func

        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = perturb_func_kwargs
    
    def run(self):
        """
        Compute the Relative Output Stability metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes them
        to :class:`quantus.RelativeOutputStability`. The model is set to evaluation
        mode before computing the metric, and ``ctx.device`` is forwarded to
        Quantus when available.

        Returns
        -------
        List[float]
            Relative Output Stability score for each evaluated observation. Lower
            values indicate more stable explanations with respect to changes in the
            model output.

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

        ctx.model.eval()

        results = quantus.RelativeOutputStability(
            nr_samples=nr_samples,
            abs=abs_,
            normalise=normalise,
            normalise_func=self.normalise_func,
            normalise_func_kwargs=self.normalise_func_kwargs,
            perturb_func=self.perturb_func,
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