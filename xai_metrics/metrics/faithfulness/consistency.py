# XAI_metrics/metrics/faithfulness/consistency.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any, Callable, Dict

@register_metric
class Consistency(BaseMetric):
    """
    Quantus Consistency metric.

    This metric evaluates whether similar explanations are assigned to
    observations with the same predicted class. Continuous attribution vectors
    are first discretised using ``discretise_func``. Then, the metric compares
    the model predictions of observations that share the same discretised
    explanation label.

    The metric is based on the Consistency metric proposed by Dasgupta et al.
    (2022) and implemented in Quantus.
    """
    NAME = 'Consistency'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
        discretise_func: Callable[[Any], Any] | None = None,
        normalise_func: Callable[..., np.ndarray] | None = None,
        normalise_func_kwargs: Dict[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, ``y_test``, selected observations and attribution values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``True``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            If ``None``, an empty dictionary is used.
        discretise_func : Callable[[Any], Any] or None, optional
            Function used to discretise continuous attribution vectors before
            comparing explanations. The function must accept one attribution vector
            as input and return a discrete representation, such as a hashable label
            or encoded pattern. If ``None``, Quantus uses its default
            discretisation function.
        normalise_func : Callable[..., numpy.ndarray] or None, optional
            Custom normalisation function passed to Quantus. The function must
            accept the attribution array as its first argument and may accept
            additional keyword arguments from ``normalise_func_kwargs``. If
            ``None``, Quantus uses its default normalisation behaviour when
            ``normalise=True``.
        normalise_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``normalise_func`` when normalisation is
            enabled. If ``None``, no additional keyword arguments are passed.
        """
        super().__init__(context, params)
        self.discretise_func = discretise_func
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs
    
    
    def run(self):
        """
        Compute the Consistency metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes them
        to :class:`quantus.Consistency`. The model is set to evaluation mode before
        computing the metric.

        Returns
        -------
        List[float]
            Consistency score for each evaluated observation, as returned by
            Quantus.

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

        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", True))

        ctx.model.eval()

        results = quantus.Consistency(
            discretise_func=self.discretise_func,
            abs=abs_,
            normalise=normalise,
            normalise_func=self.normalise_func,
            normalise_func_kwargs=self.normalise_func_kwargs
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=ctx.attributions
        )

        return results
