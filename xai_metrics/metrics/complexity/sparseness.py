# XAI_metrics/metrics/complexity/sparseness.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Mapping, Any, Callable, Dict

@register_metric
class Sparseness(BaseMetric):
    """
    Quantus Sparseness metric.

    This metric evaluates the sparsity of an explanation by computing the Gini
    Index over the absolute attribution values. Sparse explanations assign high
    importance to a small subset of features and low or negligible importance
    to the remaining ones.

    The metric is based on the Sparseness metric proposed by Chalasani et al.
    (2020) and implemented in Quantus.
    """
    NAME = 'Sparseness'

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
        Compute the Sparseness metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes
        them to :class:`quantus.Sparseness`.

        If all attribution values are negative, their absolute values are used
        before calling Quantus.

        Returns
        -------
        List[float]
            Sparseness score for each evaluated observation. Higher values
            indicate sparser explanations.
        """
        ctx = self.context
        p = self.params

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            attributions = np.abs(attributions)

        normalise = bool(p.get("normalise", True))
        
        ctx.model.train()

        results = quantus.Sparseness(
            normalise=normalise,
            normalise_func=self.normalise_func,
            normalise_func_kwargs=self.normalise_func_kwargs
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations],
            y_batch=ctx.y_test.loc[ctx.observations],
            a_batch=attributions
        )

        return results