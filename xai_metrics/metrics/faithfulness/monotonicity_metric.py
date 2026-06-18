# xai_metrics/metrics/faithfulness/monotonicity_metric.py
import numpy as np
import pandas as pd
from aix360.metrics import monotonicity_metric

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Any, Mapping


@register_metric
class MonotonicityMetric(BaseMetric):
    """
    AIX360 Monotonicity metric.

    This metric evaluates whether the probability assigned to the originally
    predicted class increases monotonically as features are progressively
    restored from a baseline input.

    For each observation, features are restored in increasing order of their
    attribution values. The metric returns ``True`` when the resulting sequence
    of predicted-class probabilities is monotonically non-decreasing.

    The wrapped AIX360 implementation requires a classification model exposing
    a ``predict_proba`` method.

    The metric is based on the monotonicity criterion described by Luss et al.
    (2019) and implemented in AIX360.
    """
    NAME = "MonotonicityMetric"

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
            The model must implement ``predict_proba``.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``base_values`` : array-like or None, optional
              Explicit baseline values used as the initial feature values. If
              provided, this takes priority over ``base_func`` and
              ``base_strategy``.

            - ``base_strategy`` : str, optional
              Strategy used to compute baseline values from ``X_test`` when
              ``base_values`` and ``base_func`` are not provided. Supported
              values are ``"mean"``, ``"median"`` and ``"zero"``. The default
              value is ``"mean"``.

            If ``None``, an empty dictionary is used.
        """
        super().__init__(context, params)


    def run(self):
        """
        Compute the Monotonicity metric.

        The method resolves a common baseline vector from the complete test
        dataset and evaluates each selected observation independently using
        :func:`aix360.metrics.monotonicity_metric`.

        For each observation, AIX360 determines the class predicted for the
        original input. Starting from the baseline, it restores features
        cumulatively in increasing order of attribution value and evaluates the
        probability assigned to that class after each restoration.

        Returns
        -------
        List[bool]
            Monotonicity result for each evaluated observation. ``True``
            indicates that the predicted-class probability never decreases as
            features are progressively restored.

        Raises
        ------
        ValueError
            If ``base_strategy`` is not ``"mean"``, ``"median"`` or
            ``"zero"``.
        AttributeError
            If the model does not implement the ``predict_proba`` method
            required by AIX360.
        """
        ctx = self.context
        p = self.params

        model = ctx.model
        X_selected = ctx.X_test.loc[ctx.observations]
        base = self._resolve_base(
            X_reference=ctx.X_test,
            base_values=p.get("base_values"),
            base_strategy=p.get("base_strategy", "mean")
        )

        scores = []
        for x_row, coefs in zip(X_selected.values, ctx.attributions):
            score = monotonicity_metric(
                model=model,
                x=np.asarray(x_row, dtype=float),
                coefs=np.asarray(coefs, dtype=float),
                base=base,
            )
            scores.append(bool(score))

        return scores

    @staticmethod
    def _resolve_base(
        X_reference: pd.DataFrame | np.ndarray,
        base_values: np.ndarray | None = None,
        base_strategy: str = "mean"
    ) -> np.ndarray:
        """
        Resolve the baseline vector used by the Monotonicity metric.

        Explicit baseline values are used when provided. Otherwise, the
        baseline is computed feature-wise from the reference dataset using the
        selected strategy.

        Parameters
        ----------
        X_reference : pandas.DataFrame or numpy.ndarray
            Reference dataset used to compute the baseline. Rows represent
            observations and columns represent input features.
        base_values : array-like or None, optional
            Explicit baseline containing one value per feature. If provided,
            it is converted to a floating-point NumPy array and returned
            directly.
        base_strategy : str, default="mean"
            Strategy used when ``base_values`` is not provided:

            - ``"mean"``: feature-wise mean of the reference dataset.
            - ``"median"``: feature-wise median of the reference dataset.
            - ``"zero"``: vector of zeros.

        Returns
        -------
        numpy.ndarray
            One-dimensional baseline array containing one value per feature.

        Raises
        ------
        ValueError
            If ``base_strategy`` is not ``"mean"``, ``"median"`` or
            ``"zero"``.
        """
        if base_values is not None:
            return np.asarray(base_values, dtype=float)

        values = (
            X_reference.values
            if hasattr(X_reference, "values")
            else np.asarray(X_reference, dtype=float)
        )

        if base_strategy == "mean":
            return np.mean(values, axis=0)
        if base_strategy == "median":
            return np.median(values, axis=0)
        if base_strategy == "zero":
            return np.zeros(values.shape[1], dtype=float)

        raise ValueError(f"Unknown base_strategy: {base_strategy}")
