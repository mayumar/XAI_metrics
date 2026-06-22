# xai_metrics/explainers/breakdown.py
import dalex as dx
import numpy as np
import pandas as pd
from torch import Tensor

from xai_metrics.base import register_explainer, BaseExplainer
from xai_metrics.base.base_explainer import ExplainerContext

from typing import Any, Mapping, Tuple, Callable

@register_explainer
class BreakDownExplainer(BaseExplainer):
    """
    DALEX Break Down explainer.

    This explainer computes local feature attributions using DALEX
    ``predict_parts`` with ``type="break_down"``. For each observation, the
    DALEX contributions are converted into a fixed-size attribution vector
    whose order follows the columns of ``context.X_background``.

    The explainer supports classification and regression models through a
    configurable prediction function. For classification, the selected output
    column of ``predict_proba`` is used.

    The explainer is registered under the name ``"BreakDown"``.
    """
    NAME = "BreakDown"

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
            contain ``y_background``. The background data columns define the
            order of the returned attribution vectors.
        params : Mapping[str, Any] or None, optional
            Explainer-specific parameters. Supported keys are:

            - ``mode`` : str, optional
              Model task type. If ``"classification"``, the explainer uses
              ``predict_proba``. Otherwise, it uses ``predict``. The default
              value is ``"classification"``.

            - ``output_index`` : int, optional
              Output column selected from ``predict_proba`` for classification
              models. The default value is ``1``.

            - ``label`` : str, optional
              Label passed to the DALEX explainer. If not provided, the model
              class name is used.

            - ``verbose`` : bool, optional
              Whether DALEX should print additional information. The default
              value is ``False``.

            - ``precalculate`` : bool, optional
              Whether DALEX should precalculate model diagnostics. The default
              value is ``True``.

            - ``order`` : Any, optional
              Feature order passed to ``predict_parts``.

            - ``n_samples`` : int or None, optional
              Number of samples passed to ``predict_parts`` through ``N``.

            - ``keep_distributions`` : bool, optional
              Whether DALEX should keep contribution distributions. The default
              value is ``False``.

            - ``random_state`` : int, optional
              Random state passed to ``predict_parts``. The default value is
              ``42``.

            - ``abs`` : bool, optional
              Whether to return absolute attribution values. The default value
              is ``False``.

            If ``None``, an empty dictionary is used.
        """
        super().__init__(context, params)
        
        self.cols = list(context.X_background.columns)
        self._explainers = {}


    def _to_dataframe(self, inputs: Any) -> pd.DataFrame:
        """
        Convert input data to a pandas dataframe.

        Parameters
        ----------
        inputs : Any
            Input observation or batch of observations. Supported inputs include
            pandas dataframes, torch tensors and array-like objects.

        Returns
        -------
        pandas.DataFrame
            Input data represented as a dataframe with the same columns as
            ``context.X_background``.
        """
        if isinstance(inputs, pd.DataFrame):
            return inputs.loc[:, self.cols]

        if isinstance(inputs, Tensor):
            values = inputs.detach().cpu().numpy()
        else:
            values = np.asarray(inputs)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        return pd.DataFrame(values, columns=self.cols)


    def _get_predict_fn(self, model: Any, predict_fn: Callable | None = None) -> Callable:
        """
        Resolve the prediction function used by DALEX.

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
            Prediction function compatible with DALEX.

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
            method = getattr(model, method_name)
        elif hasattr(model, "model") and hasattr(model.model, method_name):
            method = getattr(model.model, method_name)
        else:
            raise AttributeError(
                f"No se encontro {method_name} en {type(model)} ni en model.model"
            )
        
        if mode == "classification":
            output_index = int(self.params.get("output_index", 1))

            def predict_function(m, data):
                values = np.asarray(method(data))
                if values.ndim == 2:
                    return values[:, output_index]
                return values
            
            return predict_function
            
        def predict_function(m, data):
            return np.asarray(method(data))
        
        return predict_function
    

    def _result_to_weights(self, result: pd.DataFrame) -> np.ndarray:
        """
        Convert a DALEX result dataframe into an attribution vector.

        DALEX returns feature contributions in a dataframe. This method extracts
        the feature names, ignores auxiliary rows and accumulates the
        contribution of each feature according to the column order stored in
        ``self.cols``.

        Parameters
        ----------
        result : pandas.DataFrame
            Result dataframe returned by DALEX ``predict_parts``.

        Returns
        -------
        numpy.ndarray
            One-dimensional attribution vector with one value per input feature.
        """
        weights = np.zeros(len(self.cols), dtype=float)

        for _, row in result.iterrows():
            name = row.get("variable_name", row.get("variable"))
            if name is None or str(name).startswith("_"):
                continue

            if name in self.cols:
                feature_name = name
            else:
                feature_name = str(name).split(" = ")[0]
            
            if feature_name in self.cols:
                idx = self.cols.index(feature_name)
                weights[idx] += float(row['contribution'])

        return weights
    

    def _build_explainer(self, model: Any) -> dx.Explainer:
        """
        Build a DALEX explainer for the given model.

        Parameters
        ----------
        model : Any
            Model to wrap with DALEX.

        Returns
        -------
        dalex.Explainer
            DALEX explainer configured with the background data, targets and
            prediction function stored in this explainer.
        """
        return dx.Explainer(
            model=model,
            data=self.context.X_background,
            y=self.context.y_background,
            predict_function=self._get_predict_fn(model),
            label=self.params.get("label", type(model).__name__),
            verbose=self.params.get("verbose", False),
            precalculate=self.params.get("precalculate", True),
            model_type=self.params.get("mode", "classification")
        )
    

    def _get_explainer(self, model: Any) -> dx.Explainer:
        """
        Return a cached DALEX explainer for the given model.

        A separate DALEX explainer is cached for each model object using the
        model identity. If no valid cached explainer exists, a new one is built.

        Parameters
        ----------
        model : Any
            Model to explain.

        Returns
        -------
        dalex.Explainer
            Cached or newly created DALEX explainer.
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
        Generate Break Down attributions for the provided inputs.

        The method converts the input data to a dataframe, obtains a cached
        DALEX explainer for the model and computes one ``break_down``
        explanation per observation. The DALEX contributions are converted into
        attribution vectors ordered according to ``context.X_background``.

        Parameters
        ----------
        model : Any
            Model whose predictions are to be explained.
        inputs : Any
            Input observation or batch of observations to explain.
        targets : Any or None, optional
            Unused by this explainer. The target output is controlled through
            the prediction function and, in classification mode, through
            ``output_index``.
        **kwargs : Any
            Additional keyword arguments. They are accepted for compatibility
            with the common explainer interface.

        Returns
        -------
        numpy.ndarray
            Attribution matrix with one row per explained observation and one
            column per feature. If ``abs=True``, absolute attribution values are
            returned.
        """
        X_df = self._to_dataframe(inputs)

        explainer = self._get_explainer(model)

        attributions = []
        for _, row in X_df.iterrows():
            explanation = explainer.predict_parts(
                new_observation=row.to_frame().T,
                type="break_down",
                order=self.params.get("order"),
                N=self.params.get("n_samples"),
                keep_distributions=self.params.get("keep_distributions", False),
                random_state=self.params.get("random_state", 42)
            )

            attributions.append(self._result_to_weights(explanation.result)) # type: ignore

        values = np.asarray(attributions)

        if self.params.get("abs", False):
            values = np.abs(values)

        return values