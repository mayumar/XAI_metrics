# xai_metrics/base/base_metric.py
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
import torch.nn as nn
import pandas as pd

@dataclass(frozen=True)
class MetricContext:
    """
    Shared context used by metric implementations.

    This dataclass stores the model, test data, labels, selected observations,
    attribution values, and optional extra information required by metric
    classes.

    Attributes
    ----------
    model : torch.nn.Module
        Model evaluated by the metrics.
    X_test : pandas.DataFrame
        Test input data used for metric computation.
    y_test : pandas.Series
        Test labels associated with ``X_test``.
    observations : Any
        Identifiers of the observations explained by the attribution matrix.
        These values usually correspond to indexes in ``X_test`` and
        ``y_test``.
    attributions : numpy.ndarray
        Attribution values for the selected observations. Each row usually
        corresponds to one observation and each column to one feature.
    device : str or None
        Device where the model is placed, such as ``"cpu"`` or ``"cuda"``.
        If ``None``, no explicit device was configured.
    """
    model: nn.Module
    X_test: pd.DataFrame
    y_test: pd.Series
    observations: Any
    attributions: np.ndarray
    device: str | None = None


class MetricSkipped(Exception):
    """
    Exception raised when a metric cannot be computed.

    This exception should be used when a metric is not applicable to the
    current context, model, data, or attribution values.
    """


class BaseMetric:
    """
    Base class for all metric implementations.

    Metric classes should inherit from this class, define their own ``NAME``,
    and implement the :meth:`run` method.
    """
    NAME: str = 'metric'

    def __init__(
        self,
        context: MetricContext,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : MetricContext
            Shared context containing the model, test data, labels,
            observations, attribution values and optional device information.
        params : Mapping[str, Any] or None, optional
            Metric-specific configuration parameters. If ``None``, an empty
            dictionary is used.
        """
        self.context = context
        self.params = dict(params or {})
    
    def run(self):
        """
        Execute the metric computation.

        Subclasses must override this method with the actual metric logic.

        Returns
        -------
        Any
            Metric result returned by the concrete implementation.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("This class does not implement a run method")