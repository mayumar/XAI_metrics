# XAI_metrics/metrics/faithfulness/monotonicity_correlation.py
import quantus
import numpy as np
from scipy.stats import spearmanr

from xai_metrics.base import BaseMetric, MetricContext, register_metric, MetricSkipped

from typing import Mapping, Any, Callable, Dict

@register_metric
class MonotonicityCorrelation(BaseMetric):
    """
    Quantus Monotonicity Correlation metric.

    This metric evaluates whether features with larger attribution values have
    a stronger and more monotonic effect on the model output when they are
    perturbed. It computes the correlation between attribution values and the
    output variation caused by perturbing groups of features.

    By default, this wrapper uses a safe Spearman correlation implementation
    that returns ``0.0`` when one of the compared vectors has zero variance.

    The metric is based on the Monotonicity Correlation metric proposed by
    Nguyen and Rodríguez Martínez (2020) and implemented in Quantus.
    """
    NAME = 'MonotonicityCorrelation'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
        similarity_func: Callable[..., float | np.ndarray] | None = None,
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
        similarity_func : Callable[..., float | numpy.ndarray] or None, optional
            Function used to compare attribution values with output variation. The
            function must accept ``a`` and ``b`` as inputs and may accept
            ``batched`` and other keyword arguments. If ``None``,
            :meth:`_safe_spearman` is used.
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
        self.similarity_func = similarity_func or self._safe_spearman
        self.normalise_func = normalise_func
        self.normalise_func_kwargs = normalise_func_kwargs
        self.perturb_func = perturb_func
        self.perturb_func_kwargs = perturb_func_kwargs

    def _safe_spearman(
        self,
        a: Any,
        b: Any,
        batched: bool = False,
        **kwargs: Any
    ) -> float | np.ndarray:
        """
        Compute Spearman correlation safely.

        This helper avoids undefined Spearman correlations when one of the input
        vectors has zero variance. In that case, it returns ``0.0`` instead of
        ``nan``.

        Parameters
        ----------
        a : Any
            First input array or batch of arrays.
        b : Any
            Second input array or batch of arrays.
        batched : bool, default=False
            Whether ``a`` and ``b`` contain batches of vectors. If ``True``, the
            Spearman correlation is computed independently for each pair of
            vectors.
        **kwargs : Any
            Additional keyword arguments. These are accepted for compatibility
            with Quantus similarity functions and are not used.

        Returns
        -------
        float or numpy.ndarray
            Spearman correlation score. If ``batched=False``, a single float is
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
                    scores.append(spearmanr(ai, bi).correlation)
            return np.asarray(scores)

        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0

        return spearmanr(a, b).correlation

    def run(self):
        """
        Compute the Monotonicity Correlation metric.

        The method selects the observations defined in the metric context,
        retrieves their input data, labels and attribution values, and passes them
        to :class:`quantus.MonotonicityCorrelation`. The model is set to evaluation
        mode before computing the metric.

        Returns
        -------
        List[float]
            Monotonicity Correlation score for each evaluated observation. Higher
            values indicate a stronger monotonic relationship between attribution
            importance and model output variation after perturbation.

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

        eps = float(p.get("eps", 1e-5))
        nr_samples = int(p.get("nr_samples", 100))
        features_in_step = int(p.get("features_in_step", 1))
        abs_ = bool(p.get("abs", True))
        normalise = bool(p.get("normalise", True))
        perturb_baseline = str(p.get("perturb_baseline", "uniform"))

        ctx.model.eval()

        results = quantus.MonotonicityCorrelation(
            similarity_func=self.similarity_func,
            eps=eps,
            nr_samples=nr_samples,
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
