# xai_metrics/explainers/shap.py
import shap
import numpy as np
import pandas as pd
from torch import Tensor

from xai_metrics.base import register_explainer, BaseExplainer
from xai_metrics.base.base_explainer import ExplainerContext

from typing import Any, Mapping, Callable

@register_explainer
class SHAPExplainer(BaseExplainer):
    """
    SHAP explainer.

    This explainer computes feature attributions using :class:`shap.Explainer`.
    Background data from ``context.X_background`` are used to initialise the
    SHAP explainer. For each batch of inputs, the SHAP values are converted into
    an attribution matrix whose columns follow the order of the background data.

    For multi-output explanations, the output to explain can be selected through
    ``output_index``, through the provided ``targets`` or through the context
    batch labels. If none of these are available, ``default_output`` is used.

    The explainer is registered under the name ``"SHAP"``.
    """
    NAME = "SHAP"

    def __init__(
        self,
        context: ExplainerContext,
        params: Mapping[str, Any] | None = None
    ):
        """
        Parameters
        ----------
        context : ExplainerContext
            Shared explainer context. It must contain ``X_background`` and may
            contain ``y_batch``. The background data are used to initialise the
            SHAP explainer, and their columns define the order of the returned
            attribution vectors.
        params : Mapping[str, Any] or None, optional
            Explainer-specific parameters. Supported keys are:

            - ``mode`` : str, optional
              Model task type. If ``"classification"``, the explainer uses
              ``predict_proba``. Otherwise, it uses ``predict``. The default
              value is ``"classification"``.

            - ``max_background_samples`` : int or None, optional
              Maximum number of background observations used by SHAP. If
              provided, the background data are subsampled with
              :func:`shap.sample`.

            - ``random_state`` : int, optional
              Random state used for background sampling and SHAP explainer
              initialisation. The default value is ``42``.

            - ``algorithm`` : str, optional
              SHAP algorithm passed to :class:`shap.Explainer`. The default
              value is ``"auto"``.

            - ``output_names`` : list or None, optional
              Output names passed to :class:`shap.Explainer`.

            - ``output_index`` : int or None, optional
              Output index used when SHAP returns multi-output values. If
              provided, it takes precedence over ``targets`` and
              ``context.y_batch``.

            - ``default_output`` : int, optional
              Output index used when SHAP returns multi-output values and no
              explicit output index or targets are available. The default value
              is ``1``.

            - ``max_evals`` : int or None, optional
              Maximum number of evaluations passed to SHAP. If ``None``,
              ``"auto"`` is used.

            - ``batch_size`` : int or None, optional
              Batch size passed to SHAP. If ``None``, ``"auto"`` is used.

            - ``abs`` : bool, optional
              Whether to return absolute SHAP values. The default value is
              ``False``.

            If ``None``, an empty dictionary is used.
        """
        super().__init__(context, params)

        self.cols = list(context.X_background.columns)
        self.background = self._to_numpy(self.context.X_background)

        max_background_samples = self.params.get("max_background_samples")
        if max_background_samples is not None:
            self.background = shap.sample(
                self.background,
                int(max_background_samples),
                random_state=self.params.get("random_state", 42)
            )

        self._explainers = {}


    def _to_numpy(self, inputs: Any) -> np.ndarray:
        """
        Convert input data to a NumPy array.

        Parameters
        ----------
        inputs : Any
            Input observation or batch of observations. Supported inputs include
            pandas dataframes, torch tensors and array-like objects.

        Returns
        -------
        numpy.ndarray
            Two-dimensional array with one row per observation and columns in
            the same order as ``context.X_background``.
        """
        if isinstance(inputs, pd.DataFrame):
            values = inputs.loc[:, self.cols].to_numpy()
        elif isinstance(inputs, Tensor):
            values = inputs.detach().cpu().numpy()
        else:
            values = np.asarray(inputs)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        return values


    def _get_predict_fn(self, model: Any, predict_fn: Callable | None = None) -> Callable:
        """
        Resolve the prediction function used by SHAP.

        If ``predict_fn`` is provided, it is returned directly. Otherwise, the
        method searches for ``predict_proba`` in classification mode and
        ``predict`` in any other mode. The method is searched first on the model
        itself and then on ``model.model``.

        Parameters
        ----------
        model : Any
            Model to explain.
        predict_fn : Callable or None, optional
            Custom prediction function. If provided, it is used directly.

        Returns
        -------
        Callable
            Prediction function used by SHAP.

        Raises
        ------
        AttributeError
            If the required prediction method is not available on ``model`` or
            ``model.model``.
        """
        if predict_fn is not None:
            return predict_fn

        mode = self.params.get("mode", "classification")

        if mode == "classification":
            method_name = "predict_proba"
        else:
            method_name = "predict"

        if hasattr(model, method_name):
            return getattr(model, method_name)

        if hasattr(model, "model") and hasattr(model.model, method_name):
            return getattr(model.model, method_name)

        raise AttributeError(
            f"No se encontro {method_name} en {type(model)} ni en model.model"
        )
    

    def _select_values(
        self,
        shap_values: np.ndarray,
        targets: Any | None
    ) -> np.ndarray:
        """
        Select the SHAP values used as attributions.

        Two-dimensional SHAP values are returned directly. For multi-output
        values, the selected output is determined using ``output_index`` first,
        then the provided ``targets``, then ``context.y_batch`` and finally
        ``default_output``.

        Parameters
        ----------
        shap_values : numpy.ndarray
            SHAP values returned by the SHAP explainer.
        targets : Any or None
            Target labels used to select the output dimension when SHAP returns
            multi-output values.

        Returns
        -------
        numpy.ndarray
            Two-dimensional attribution matrix with one row per observation and
            one column per feature.

        Raises
        ------
        ValueError
            If the SHAP values do not have two or three dimensions, or if the
            target length does not match the number of explained samples.
        """
        if shap_values.ndim == 2:
            return shap_values

        if shap_values.ndim != 3:
            raise ValueError(
                "Unexpected SHAP values shape. "
                f"Expected 2D or 3D array, got {shap_values.shape}."
            )

        output_index = self.params.get("output_index")

        if output_index is not None:
            return shap_values[:, :, int(output_index)]
        
        if targets is None:
            targets = self.context.y_batch
        
        if targets is not None:
            targets = np.asarray(targets).reshape(-1).astype(int)

            if len(targets) != shap_values.shape[0]:
                raise ValueError(
                    "targets length must match the number of explained samples."
                )
            
            return np.asarray([
                shap_values[i, :, targets[i]]
                for i in range(shap_values.shape[0])
            ])
        
        default_output = int(self.params.get("default_output", 1))
        return shap_values[:, :, default_output]


    def _build_explainer(self, model: Any) -> shap.Explainer:
        """
        Build a SHAP explainer for the given model.

        Parameters
        ----------
        model : Any
            Model to explain.

        Returns
        -------
        shap.Explainer
            SHAP explainer configured with the model prediction function,
            background data and feature names.
        """
        predict_fn = self._get_predict_fn(model)

        return shap.Explainer(
            predict_fn,
            self.background,
            algorithm=self.params.get("algorithm", "auto"),
            output_names=self.params.get("output_names"),
            feature_names=self.cols,
            seed=self.params.get("random_state", 42)
        )


    def _get_explainer(self, model: Any) -> shap.Explainer:
        """
        Return a cached SHAP explainer for the given model.

        A separate SHAP explainer is cached for each model object using the
        model identity. If no valid cached explainer exists, a new one is built.

        Parameters
        ----------
        model : Any
            Model to explain.

        Returns
        -------
        shap.Explainer
            Cached or newly created SHAP explainer.
        """
        key = id(model)
        cached = self._explainers.get(key)

        if cached is not None and cached[0] is model:
            return cached[1]

        explainer = self._build_explainer(model)
        self._explainers[key] = (model, explainer)

        return explainer
    

    def explain(
        self,
        model: Any,
        inputs: Any,
        targets: Any | None = None,
        **kwargs: Any
    ) -> np.ndarray:
        """
        Generate SHAP attributions for the provided inputs.

        The method converts the inputs to a NumPy array, obtains a cached SHAP
        explainer and computes SHAP values for the batch. Multi-output values
        are reduced to a single output according to ``output_index``,
        ``targets``, ``context.y_batch`` or ``default_output``.

        Parameters
        ----------
        model : Any
            Model whose predictions are to be explained.
        inputs : Any
            Input observation or batch of observations to explain.
        targets : Any or None, optional
            Target labels used to select the output dimension when SHAP returns
            multi-output values.
        **kwargs : Any
            Additional keyword arguments. They are accepted for compatibility
            with the common explainer interface.

        Returns
        -------
        numpy.ndarray
            Attribution matrix with one row per explained observation and one
            column per feature. If ``abs=True``, absolute SHAP values are
            returned.
        """
        X_np = self._to_numpy(inputs)

        explainer = self._get_explainer(model)

        max_evals = self.params.get("max_evals")
        max_evals = int(max_evals) if max_evals is not None else "auto"

        batch_size = self.params.get("batch_size")
        batch_size = int(batch_size) if batch_size is not None else "auto"

        explanation = explainer(
            X_np,
            max_evals=max_evals,
            batch_size=batch_size
        )

        values = np.asarray(explanation.values) # type: ignore

        values = self._select_values(
            shap_values=values,
            targets=targets
        )

        if self.params.get("abs", False):
            values = np.abs(values)

        return values