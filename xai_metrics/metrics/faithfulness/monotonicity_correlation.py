# xai_metrics/metrics/faithfulness/monotonicity_correlation.py
import quantus
import numpy as np
from scipy.stats import spearmanr

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any

@register_metric
class MonotonicityCorrelation(BaseMetric):
    """
    Quantus Monotonicity Correlation metric.

    This metric evaluates whether feature attribution values are monotonically
    related to the uncertainty caused by perturbing the corresponding features.
    Features are grouped in increasing order of attribution, and each group is
    perturbed repeatedly to estimate its effect on the target model output. The
    score is the Spearman correlation between the attribution sums and the
    estimated output variations.

    This wrapper uses a safe Spearman correlation that returns ``0.0`` when
    either input vector has zero variance.

    Higher scores indicate a stronger positive relationship between feature
    importance and the uncertainty caused by perturbing those features.

    The metric is based on the Monotonicity Correlation metric proposed by
    Nguyen and Rodríguez Martínez (2020) and implemented in Quantus.
    """
    NAME = 'MonotonicityCorrelation'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared metric evaluation context. It must contain the model,
            ``X_test``, ``y_test``, selected observations and attribution values.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``eps`` : float, optional
              Threshold used when computing the inverse prediction factor. The
              default value is ``1e-5``.

            - ``nr_samples`` : int, optional
              Number of perturbation samples generated for each feature group.
              The default value is ``100``.

            - ``features_in_step`` : int, optional
              Number of features perturbed at each step. The default value is
              ``1``.

            - ``abs`` : bool, optional
              Whether to apply the absolute value operation to the attribution
              values before computing the metric. The default value is ``True``.

            - ``normalise`` : bool, optional
              Whether to normalise the attribution values before computing the
              metric. The default value is ``True``.

            - ``perturb_baseline`` : str, optional
              Baseline value used when perturbing features. Supported values
              depend on the Quantus perturbation function. Common values are
              ``"black"``, ``"white"``, ``"mean"``, ``"random"`` and
              ``"uniform"``. The default value is ``"uniform"``.

            If ``None``, an empty dictionary is used.

        Notes
        -----
        The wrapper uses :meth:`_safe_spearman` as the similarity function and
        the default normalisation and perturbation functions provided by
        Quantus.
        """
        super().__init__(context, params)


    def _safe_spearman(
        self,
        a: Any,
        b: Any,
        batched: bool = False,
        **kwargs: Any
    ) -> float | np.ndarray:
        """
        Compute the Spearman correlation while handling constant inputs.

        Parameters
        ----------
        a : Any
            First vector or batch of vectors.
        b : Any
            Second vector or batch of vectors.
        batched : bool, default=False
            Whether to compute one correlation for each pair of vectors in
            ``a`` and ``b``.
        **kwargs : Any
            Additional unused arguments accepted for compatibility with
            Quantus.

        Returns
        -------
        float or numpy.ndarray
            Spearman correlation coefficient. A value of ``0.0`` is returned
            when either compared vector has zero variance.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        if batched:
            scores = []
            for ai, bi in zip(a, b):
                if np.std(ai) == 0 or np.std(bi) == 0:
                    scores.append(0.0)
                else:
                    scores.append(spearmanr(ai, bi).correlation)
            return np.asarray(scores)

        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        return spearmanr(a, b).correlation

    def run(self):
        """
        Compute the Monotonicity Correlation metric.

        The method passes the selected inputs, labels and attribution values to
        :class:`quantus.MonotonicityCorrelation`. Quantus perturbs groups of
        features repeatedly, estimates their relative effect on the target
        output and compares those estimates with the corresponding attribution
        sums using :meth:`_safe_spearman`.

        If all attribution values are negative, their absolute values are used
        when ``abs=True``; otherwise, the metric is skipped. The model is set to
        evaluation mode before the computation.

        Returns
        -------
        List[float]
            Monotonicity Correlation score for each evaluated observation.
            Higher values indicate a stronger positive relationship between
            attribution importance and output variation.

        Raises
        ------
        MetricSkipped
            If all attribution values are negative and ``abs`` is ``False``.
        """
        ctx = self.context
        p = self.params

        eps = float(p.get("eps", 1e-5))
        nr_samples = int(p.get("nr_samples", 100))
        features_in_step = int(p.get("features_in_step", 1))
        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", True))
        perturb_baseline = str(p.get("perturb_baseline", "uniform"))

        attributions = ctx.attributions
        if np.all(attributions < 0.0):
            if not abs_:
                raise MetricSkipped(
                    f"{self.NAME} skipped: all attributions are negative."
                )
            else:
                attributions = np.abs(attributions)

        ctx.model.eval()

        results = quantus.MonotonicityCorrelation(
            similarity_func=self._safe_spearman,
            eps=eps,
            nr_samples=nr_samples,
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
