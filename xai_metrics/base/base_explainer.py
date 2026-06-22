# xai_metrics/base/base_explainer.py
from dataclasses import dataclass
import numpy as np
import pandas as pd
from torch.nn import Module

from typing import Mapping, Any
from xai_metrics.base.types import ExplainFunc

@dataclass(frozen=True)
class ExplainerContext:
    """
    Shared context used by explainer implementations.

    This dataclass stores the background data required by explanation methods
    and, optionally, the model, current batch, target values and device used
    during explanation generation.

    Parameters
    ----------
    X_background : pandas.DataFrame
        Background or reference input data used by the explainer.
    y_background : pandas.Series or None, optional
        Labels or targets associated with ``X_background``. This is only
        required by explainers that need background target values.
    model : Any or None, optional
        Model associated with the explainer context. This can be used by
        explainers that store the model in the context instead of receiving it
        only at call time.
    X_batch : pandas.DataFrame or None, optional
        Current batch of observations to be explained.
    y_batch : pandas.Series or None, optional
        Labels or targets associated with ``X_batch``.
    device : str or None, optional
        Device used for model execution, such as ``"cpu"`` or ``"cuda"``.
    """
    X_background: pd.DataFrame
    y_background: pd.Series | None = None
    model: Any | None = None
    X_batch: pd.DataFrame | None = None
    y_batch: pd.Series | None = None
    device: str | None = None


class ExplainerSkipped(Exception):
    """
    Exception raised when an explainer cannot generate attributions.

    This exception should be used when an explainer is not applicable to the
    current context, model, data, targets, or configuration.
    """


class BaseExplainer:
    """
    Base class for explanation methods.

    Explainer classes should inherit from this class, define their own
    ``NAME`` and implement :meth:`explain`, which generates attribution values
    for a batch of inputs.

    Instances are callable, so calling an explainer is equivalent to calling
    :meth:`explain`. They can also expose their explanation method through
    :meth:`as_explain_func`, which returns a metric-compatible function for
    metrics that need to recompute explanations on perturbed inputs.
    """
    NAME: str = "explainer"

    def __init__(
        self,
        context: ExplainerContext,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : ExplainerContext
            Shared context containing the background data and optional model,
            batch, labels and device information used by the explainer.
        params : Mapping[str, Any] or None, optional
            Explainer-specific configuration parameters. A copy of the mapping
            is stored in :attr:`params`. If ``None``, an empty dictionary is
            used.
        """
        self.context = context
        self.params = dict(params or {})


    def explain(
        self,
        model: Module,
        inputs: Any,
        targets: Any | None = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Generate explanations for the provided inputs.
        
        Subclasses must override this method with the logic required by
        the corresponding explanation method.
        
        Parameters
        ----------
        model : torch.nn.Module
            Model whose predictions are to be explained.
        inputs : Any
            Input observation or batch of observations to explain.
        targets : Any or None, optional
            Target outputs, classes or labels for which explanations are generated. Their interpretation depends on the explainer.
        **kwargs : Any
            Additional arguments required by the concrete explanation method.
        
        Returns
        -------
        numpy.ndarray
            Attribution values generated for the provided inputs.
            
        Raises
        ------
        NotImplementedError
            Always raised by the base implementation. Subclasses must override this method.
        """
        raise NotImplementedError("This class does not implement a explain method")
    
    def __call__(
        self,
        model: Module,
        inputs: Any,
        targets: Any | None = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Generate explanations using the callable interface.

        This method delegates the computation to :meth:`explain`.

        Parameters
        ----------
        model : torch.nn.Module
            Model whose predictions are to be explained.
        inputs : Any
            Input observation or batch of observations to explain.
        targets : Any or None, optional
            Target outputs, classes or labels for which explanations are
            generated.
        **kwargs : Any
            Additional arguments forwarded to :meth:`explain`.

        Returns
        -------
        numpy.ndarray
            Attribution values generated by :meth:`explain`.
        """
        return self.explain(model=model, inputs=inputs, targets=targets, **kwargs)
    
    def as_explain_func(self) -> ExplainFunc:
        """
        Return the explanation method as a metric-compatible function.

        The returned bound method can be passed as an ``explain_func``
        dependency to metrics that need to generate explanations for modified
        or perturbed inputs.

        Returns
        -------
        ExplainFunc
            Bound :meth:`explain` method of the current explainer instance.
        """
        return self.explain