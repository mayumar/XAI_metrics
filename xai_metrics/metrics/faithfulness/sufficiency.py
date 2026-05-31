# XAI_metrics/metrics/faithfulness/sufficiency.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any, Callable, Dict

@register_metric
class Sufficiency(BaseMetric):
    """
    Quantus Sufficiency metric.

    This metric evaluates whether an explanation is sufficient to identify the
    model prediction. Two observations are considered to share a similar
    explanation when the distance between their attribution vectors is below a
    user-defined threshold. The score measures how often observations with
    similar explanations also share the same predicted class.

    The metric is based on the Sufficiency metric proposed by Dasgupta et al.
    (2022) and implemented in Quantus.
    """
    NAME = 'Sufficiency'

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
            ``X_test``, ``y_test``, selected observations and attribution values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``threshold`` : float, optional
              Maximum distance between two attribution vectors for their
              explanations to be considered similar. The default value is
              ``0.6``.

            - ``distance_func`` : str, optional
              Distance function used to compare attribution vectors. The value
              is passed to Quantus and is typically a valid SciPy distance name,
              such as ``"seuclidean"``. The default value is ``"seuclidean"``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is
              ``True``.

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
            Keyword arguments passed to ``normalise_func`` when normalisation is
            enabled. If ``None``, no additional keyword arguments are passed.
        """
        super().__init__(context, params)
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs

    def run(self):
        """
        Compute the Sufficiency metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes them
        to :class:`quantus.Sufficiency`. The model is set to evaluation mode before
        computing the metric.

        Returns
        -------
        List[float]
            Sufficiency score for each evaluated observation. Higher values
            indicate that observations with similar explanations more often share
            the same predicted class.

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

        threshold = float(p.get("threshold", 0.6))
        distance_func = str(p.get("distance_func", "seuclidean"))
        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", True))

        ctx.model.eval()

        results = quantus.Sufficiency(
            threshold=threshold,
            distance_func=distance_func,
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
