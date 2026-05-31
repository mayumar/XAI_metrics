# XAI_metrics/metrics/faithfulness/monotonicity.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any, Callable, Dict

@register_metric
class Monotonicity(BaseMetric):
    """
    Quantus Monotonicity metric.

    This metric evaluates whether adding features in order of increasing
    attribution importance leads to monotonically increasing model performance.
    For each observation, Quantus starts from a baseline input and progressively
    adds groups of features according to their attribution values. The metric
    returns whether the predicted probability for the target class increases
    monotonically across those perturbation steps.

    The metric is based on the Monotonicity metric proposed by Arya et al.
    (2019) and related monotonic attribute function work by Luss et al. (2019),
    as implemented in Quantus.
    """
    NAME = 'Monotonicity'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
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
            ``None``, Quantus uses its default perturbation function.
        perturb_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``perturb_func``. If ``None``, no
            additional keyword arguments are passed.
        """
        super().__init__(context, params)
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = perturb_func_kwargs

    def run(self):
        """
        Compute the Monotonicity metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes them
        to :class:`quantus.Monotonicity`. The model is set to evaluation mode
        before computing the metric.

        Returns
        -------
        List[bool]
            Monotonicity result for each evaluated observation. ``True`` indicates
            that the predicted probability increases monotonically as features are
            added in increasing order of attribution importance.

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

        features_in_step = int(p.get("features_in_step", 1))
        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", True))
        perturb_baseline = str(p.get("perturb_baseline", "black"))

        ctx.model.eval()

        results = quantus.Monotonicity(
            features_in_step=features_in_step,
            abs=abs_,
            normalise=normalise,
            normalise_func=self.normalise_func,
            normalise_func_kwargs=self.normalise_func_kwargs,
            perturb_func=self.perturb_func,
            perturb_baseline=perturb_baseline,
            perturb_func_kwargs=self.perturb_func_kwargs
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=ctx.attributions
        )

        return results
