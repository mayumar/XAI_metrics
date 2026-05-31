# XAI_metrics/metrics/complexity/complexity_metric.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Mapping, Any, Callable, Dict

@register_metric
class Complexity(BaseMetric):
    """
    Quantus Complexity metric.

    This metric evaluates how complex an explanation is by computing the
    entropy of the attribution distribution. Explanations with high entropy
    distribute importance across many features, while explanations with low
    entropy concentrate importance on fewer features.

    The metric is based on the Complexity metric proposed by Bhatt et al.
    (2020) and implemented in Quantus.
    """
    NAME = 'Complexity'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
        normalise_func: Callable[..., np.ndarray] | None = None,
        normalise_func_kwargs: Dict[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, ``y_test``, selected observations and attribution
            values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            If ``None``, an empty dictionary is used.
        normalise_func : Callable[..., numpy.ndarray] or None, optional
            Custom normalisation function passed to Quantus. The function must
            accept the attribution array as its first argument and may accept
            additional keyword arguments from ``normalise_func_kwargs``. If
            ``None``, Quantus uses its default normalisation behaviour when
            ``normalise=True``.
        normalise_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``normalise_func`` when normalisation
            is enabled. If ``None``, no additional keyword arguments are
            passed.
        """
        super().__init__(context, params)
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs

    def run(self):
        """
        Compute the Complexity metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes
        them to :class:`quantus.Complexity`.

        If all attribution values are negative, their absolute values are used
        before calling Quantus.

        Returns
        -------
        List[float]
            Complexity score for each evaluated observation. Lower values
            indicate less complex explanations.
        """
        ctx = self.context
        p = self.params

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            attributions = np.abs(attributions)

        normalise = bool(p.get("normalise", True))
        
        ctx.model.train()

        results = quantus.Complexity(
            normalise=normalise,
            normalise_func=self.normalise_func,
            normalise_func_kwargs=self.normalise_func_kwargs,
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations],
            y_batch=ctx.y_test.loc[ctx.observations],
            a_batch=attributions
        )

        # Normalización
        # n_features = attributions.shape[1]
        # max_entropy = np.log(n_features)
        # results = np.array(results) / max_entropy

        return results