# xai_metrics/metrics/faithfulness/faithfulness.py
import numpy as np
import pandas as pd
from aix360.metrics import faithfulness_metric

from xai_metrics.base import BaseMetric, MetricContext, register_metric

from typing import Any, Mapping


@register_metric
class Faithfulness(BaseMetric):
    """
    AIX360 Faithfulness metric.

    This metric evaluates whether feature attribution values reflect the
    influence of the corresponding features on the model prediction. For each
    observation, every feature is individually replaced with a baseline value
    and the probability assigned to the original predicted class is recorded.
    The score is the negative Pearson correlation between the attribution
    values and these probabilities.

    Higher scores indicate stronger agreement between feature importance and
    the effect of replacing features with their baseline values.

    The wrapped AIX360 implementation requires a classification model exposing
    a ``predict_proba`` method.

    The metric is based on the faithfulness criterion proposed by
    Alvarez-Melis and Jaakkola (2018) and implemented in AIX360.
    """
    NAME = "Faithfulness"

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
            ``X_test``, ``y_test``, selected observations and attribution
            values. The model must implement ``predict_proba``.
        params : Mapping[str, Any] or None, optional
            Metric-specific parameters. Supported keys are:

            - ``base_values`` : array-like or None, optional
              Explicit baseline values used to replace feature values. If
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
        Compute the Faithfulness metric.

        The method resolves a common baseline vector from the complete test
        dataset and evaluates each selected observation independently using
        :func:`aix360.metrics.faithfulness_metric`.

        For each observation, AIX360 determines the predicted class, replaces
        each feature individually with its baseline value and records the
        resulting probability for that class. The returned score is the
        negative Pearson correlation between these probabilities and the
        attribution values.

        Returns
        -------
        List[float]
            Faithfulness score for each evaluated observation. Higher values
            indicate that features with larger attribution values produce
            greater decreases in the predicted class probability when replaced.

        Raises
        ------
        ValueError
            If ``base_strategy`` is not ``"mean"``, ``"median"`` or
            ``"zero"``.
        AttributeError
            If the model does not implement the ``predict_proba`` method
            required by AIX360.

        Notes
        -----
        The score may be ``nan`` when the attribution values or the probabilities
        obtained after feature replacement have zero variance.
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
            score = faithfulness_metric(
                model=model,
                x=np.asarray(x_row, dtype=float),
                coefs=np.asarray(coefs, dtype=float),
                base=base,
            )
            scores.append(float(score))

        return scores

    @staticmethod
    def _resolve_base(
        X_reference: pd.DataFrame | np.ndarray,
        base_values: np.ndarray | None = None,
        base_strategy: str = "mean"
    ) -> np.ndarray:
        """
        Resolve the baseline vector used by the Faithfulness metric.

        Explicit baseline values are used when provided. Otherwise, one
        baseline value per feature is computed from the reference dataset
        according to the selected strategy.

        Parameters
        ----------
        X_reference : pandas.DataFrame or numpy.ndarray
            Reference dataset used to compute the baseline. Rows represent
            observations and columns represent features.
        base_values : array-like or None, optional
            Explicit baseline values containing one value per feature. If
            provided, ``base_strategy`` is ignored.
        base_strategy : str, default="mean"
            Strategy used to compute the baseline. Supported values are
            ``"mean"``, ``"median"`` and ``"zero"``.

        Returns
        -------
        numpy.ndarray
            One-dimensional floating-point array containing one baseline value
            per feature.

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
