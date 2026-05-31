# XAI_metrics/metrics/faithfulness/sensitivity_n.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any, Callable, Dict

@register_metric
class SensitivityN(BaseMetric):
    """
    Quantus Sensitivity-N metric.

    This metric evaluates whether attribution values are faithful to the model
    by comparing the sum of feature attributions with the corresponding change
    in model output after perturbing those features. Quantus computes this
    relationship over increasingly large feature subsets and compares both
    quantities using a similarity function.

    The metric is based on the Sensitivity-N test proposed by Ancona et al.
    (2018) and implemented in Quantus.
    """
    NAME = 'SensitivityN'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
        similarity_func: Callable[..., float | np.ndarray] | None = None,
        normalise_func: Callable[..., np.ndarray] | None = None,
        normalise_func_kwargs: Dict[str, Any] | None = None,
        perturb_func: Callable[[Any], Any] | None = None,
        perturb_func_kwargs: Dict[str, Any] | None = None
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
        similarity_func : Callable[..., float | numpy.ndarray] or None, optional
            Function used to compare attribution sums with prediction score drops. The
            function must accept ``a`` and ``b`` as inputs and may accept ``batched``
            and other keyword arguments. If ``None``, Quantus uses its default
            similarity function.
        normalise_func : Callable[..., numpy.ndarray] or None, optional
            Custom normalisation function passed to Quantus. The function must accept
            the attribution array as its first argument and may accept additional
            keyword arguments from ``normalise_func_kwargs``. If ``None``, Quantus uses
            its default normalisation behaviour when ``normalise=True``.
        normalise_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``normalise_func`` when normalisation is
            enabled. If ``None``, no additional keyword arguments are passed.
        perturb_func : Callable[..., numpy.ndarray] or None, optional
            Perturbation function passed to Quantus. The function must be compatible
            with Quantus perturbation functions, accepting at least an input array and
            feature indices, and returning the perturbed array. If ``None``, Quantus
            uses its default perturbation function.
        perturb_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``perturb_func``. If ``None``, no additional
            keyword arguments are passed.
        """
        super().__init__(context, params)
        self.similarity_func = similarity_func
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = perturb_func_kwargs

    def run(self):
        """
        Compute the Sensitivity-N metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes them
        to :class:`quantus.SensitivityN`. The model is set to evaluation mode
        before computing the metric.

        Returns
        -------
        List[float]
            Sensitivity-N scores as returned by Quantus. Higher values indicate
            stronger agreement between attribution sums and model output changes
            after perturbation.

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

        n_max_percentage = float(p.get("n_max_percentage", 0.8))
        features_in_step = int(p.get("features_in_step", 1))
        abs_ = bool(p.get("abs", False))
        normalise = bool(p.get("normalise", True))
        perturb_baseline = str(p.get("perturb_baseline", "black"))

        ctx.model.eval()

        results = quantus.SensitivityN(
            similarity_func=self.similarity_func,
            n_max_percentage=n_max_percentage,
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
