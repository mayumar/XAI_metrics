# XAI_metrics/metrics/faithfulness/monotonicity_metric.py
import numpy as np
import pandas as pd
from aix360.metrics import monotonicity_metric

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Any, Mapping, Callable, Dict


@register_metric
class MonotonicityMetric(BaseMetric):
    """
    AIX360 Monotonicity metric.

    This metric evaluates whether the model output changes monotonically when
    features are incrementally added according to their attribution values. For
    each observation, the input starts from a baseline vector and features are
    added one by one in increasing order of importance. The metric returns
    ``True`` when the predicted probability of the original predicted class
    increases monotonically.

    The wrapped AIX360 implementation expects a model exposing a
    ``predict_proba`` method.

    The metric is based on the monotonicity metric proposed by Luss et al.
    (2019) and implemented in AIX360.
    """
    NAME = "MonotonicityMetric"

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None,
        base_func: Callable[..., Any] | None = None,
        base_func_kwargs: Dict[str, Any] | None = None
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
        base_func : Callable[..., Any] or None, optional
            Custom function used to compute baseline values from the reference
            dataset. The function must accept ``X_reference`` as its first
            argument, usually a ``pandas.DataFrame``, and may accept additional
            keyword arguments from ``base_func_kwargs``. It must return an
            array-like object with one baseline value per feature.
        base_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``base_func``. If ``None``, no additional
            keyword arguments are passed.
        """
        super().__init__(context, params)
        self.base_func = base_func
        self.base_func_kwargs = base_func_kwargs

    def run(self):
        """
        Compute the Monotonicity metric.

        The method selects the observations defined in the metric context, resolves
        the baseline values, and computes one monotonicity result per selected
        observation using :func:`aix360.metrics.monotonicity_metric`.

        Returns
        -------
        List[bool]
            Monotonicity result for each evaluated observation. ``True`` indicates
            that the predicted probability increases monotonically as features are
            added in increasing order of attribution importance.

        Raises
        ------
        ValueError
            If ``base_strategy`` is not one of ``"mean"``, ``"median"`` or
            ``"zero"``.
        AttributeError
            If the model does not expose the ``predict_proba`` method required by
            the AIX360 implementation.
        """
        ctx = self.context
        p = self.params

        model = ctx.model
        X_selected = ctx.X_test.loc[ctx.observations]
        base = self._resolve_base(
            X_reference=ctx.X_test,
            base_values=p.get("base_values"),
            base_strategy=p.get("base_strategy", "mean"),
            base_func=self.base_func,
            base_func_kwargs=self.base_func_kwargs
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
        base_strategy: str = "mean",
        base_func: Callable[..., Any] | None = None,
        base_func_kwargs: Dict[str, Any] | None = None,
    ) -> np.ndarray:
        """
        Resolve the baseline values used by the Monotonicity metric.

        The baseline can be provided directly through ``base_values``, computed
        with a custom ``base_func``, or derived from the reference dataset using
        one of the supported baseline strategies.

        Parameters
        ----------
        X_reference : pandas.DataFrame or numpy.ndarray
            Reference dataset used to compute baseline values when ``base_values``
            and ``base_func`` are not provided.
        base_values : numpy.ndarray or None, optional
            Explicit baseline values. If provided, these values are returned as a
            NumPy array and no strategy is applied.
        base_strategy : str, default="mean"
            Strategy used to compute baseline values from ``X_reference``.
            Supported values are ``"mean"``, ``"median"`` and ``"zero"``.
        base_func : Callable[..., Any] or None, optional
            Custom function used to compute baseline values from ``X_reference``.
            The function must accept ``X_reference`` as its first argument and may
            accept additional keyword arguments from ``base_func_kwargs``. It must
            return an array-like object with one value per feature.
        base_func_kwargs : Dict[str, Any] or None, optional
            Keyword arguments passed to ``base_func``. If ``None``, an empty
            dictionary is used.

        Returns
        -------
        numpy.ndarray
            Baseline values used as the initial feature values during metric
            computation.

        Raises
        ------
        ValueError
            If ``base_strategy`` is unknown.
        """
        if base_values is not None:
            return np.asarray(base_values, dtype=float)

        if base_func is not None:
            kwargs = base_func_kwargs or {}
            return np.asarray(base_func(X_reference, **kwargs), dtype=float)

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
