# xai_metrics/metrics/faithfulness/faithfulness_estimate.py
import quantus
import numpy as np

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any


@register_metric
class FaithfulnessEstimate(BaseMetric):
    """
    Quantus Faithfulness Estimate metric.

    This metric evaluates whether features with higher attribution values
    produce larger changes in the model output when they are perturbed.
    Features are perturbed in groups ordered by attribution importance, and the
    metric compares the sum of their attributions with the corresponding drop
    in the target output.

    This wrapper uses a safe Pearson correlation function that returns ``0.0``
    when either of the compared vectors has zero variance, avoiding undefined
    correlation values.

    Higher scores indicate stronger agreement between attribution importance
    and changes in the model output.

    The metric is based on the Faithfulness Estimate proposed by Alvarez-Melis
    and Jaakkola (2018) and implemented in Quantus.
    """
    NAME = 'FaithfulnessEstimate'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None
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
              Number of features perturbed at each step. The default value is
              ``1``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``False``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            - ``perturb_baseline`` : str, optional
              Baseline value used when perturbing features. Supported values
              depend on the Quantus perturbation function. Common values are
              ``"black"``, ``"white"``, ``"mean"``, ``"random"`` and
              ``"uniform"``. The default value is ``"black"``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        This wrapper uses its internal :meth:`_safe_pearson` method as the
        similarity function. The default normalisation and perturbation
        functions provided by Quantus are used.
        """
        super().__init__(context, params)


    def _safe_pearson(
        self,
        a: Any,
        b: Any,
        batched: bool = False,
        **kwargs: Any
    ) -> float | np.ndarray:
        """
        Compute Pearson correlation safely.

        This helper avoids undefined Pearson correlations when one of the input
        vectors has zero variance. In that case, it returns ``0.0`` instead of
        ``nan``.

        Parameters
        ----------
        a : Any
            First input array or batch of arrays.
        b : Any
            Second input array or batch of arrays.
        batched : bool, default=False
            Whether ``a`` and ``b`` contain batches of vectors. If ``True``,
            the Pearson correlation is computed independently for each pair of
            vectors.
        **kwargs : Any
            Additional keyword arguments. These are accepted for compatibility
            with Quantus similarity functions and are not used.

        Returns
        -------
        float or numpy.ndarray
            Pearson correlation score. If ``batched=False``, a single float is
            returned. If ``batched=True``, a NumPy array with one score per input
            pair is returned.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if batched:
            scores = []
            for ai, bi in zip(a, b):
                if np.std(ai) == 0 or np.std(bi) == 0:
                    scores.append(0.0)
                else:
                    scores.append(np.corrcoef(ai, bi)[0, 1])
            return np.asarray(scores)

        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        return np.corrcoef(a, b)[0, 1]

    def run(self):
        """
        Compute the Faithfulness Estimate metric.

        The method selects the observations defined in the metric context and
        passes their input data, labels and attribution values to
        :class:`quantus.FaithfulnessEstimate`. Features are perturbed in groups,
        and the internal safe Pearson correlation function compares attribution
        sums with the resulting target-output drops.

        If all attribution values are negative, their treatment depends on the
        ``abs`` parameter. Their absolute values are used when ``abs=True``;
        otherwise, the metric is skipped.

        The model is set to evaluation mode before the metric is computed.

        Returns
        -------
        List[float]
            Faithfulness Estimate score for each evaluated observation. Higher
            values indicate stronger agreement between attribution importance
            and model output changes after perturbation.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        features_in_step = int(p.get("features_in_step", 1))
        abs_ = bool(p.get("abs", False))
        normalise = bool(p.get("normalise", True))
        perturb_baseline = str(p.get("perturb_baseline", "black"))

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.eval()

        results = quantus.FaithfulnessEstimate(
            similarity_func=self._safe_pearson,
            features_in_step=features_in_step,
            abs=abs_,
            normalise=normalise,
            perturb_baseline=perturb_baseline
        )(
            model=ctx.model,
            x_batch=ctx.X_test.loc[ctx.observations].to_numpy(copy=True),
            y_batch=ctx.y_test.loc[ctx.observations].to_numpy(copy=True),
            a_batch=attributions
        )

        return results
