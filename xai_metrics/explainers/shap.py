# xai_metrics/explainers/shap.py
import shap
from torch.nn import Module
import numpy as np
import pandas as pd
from torch import Tensor

from xai_metrics.base import BaseExplainer
from xai_metrics.base.base_explainer import ExplainerContext

from typing import Any, Mapping

class SHAPExplainer(BaseExplainer):
    NAME = "SHAP"

    def __init__(
        self,
        context: ExplainerContext,
        params: Mapping[str, Any] | None = None
    ):
        super().__init__(context, params)

        if context.model is None:
            raise ValueError("SHAPExplainer requires context.model.")

        self.predict_fn = self._get_predict_fn(context.model)

        self.cols = list(context.X_background.columns)

        background = self._to_numpy(self.context.X_background)

        max_background_samples = self.params.get("max_background_samples")
        if max_background_samples is not None:
            background = shap.sample(
                background,
                int(max_background_samples),
                random_state=self.params.get("random_state", 42)
            )

        self.explainer = shap.Explainer(
            self.predict_fn,
            background,
            algorithm=self.params.get("algorithm", "auto"),
            output_names=self.params.get("output_names"),
            feature_names=self.cols,
            seed=self.params.get("random_state", 42)
        )


    def _to_numpy(self, inputs: Any) -> np.ndarray:
        if isinstance(inputs, pd.DataFrame):
            values = inputs.loc[:, self.cols].to_numpy()
        elif isinstance(inputs, Tensor):
            values = inputs.detach().cpu().numpy()
        else:
            values = np.asarray(inputs)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        return values


    def _get_predict_fn(self, model: Module, predict_fn=None):
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
    

    def explain(
        self,
        model: Module,
        inputs: Any,
        targets: Any | None = None,
        **kwargs: Any
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        X_np = self._to_numpy(inputs)

        max_evals = self.params.get("max_evals")
        max_evals = int(max_evals) if max_evals is not None else "auto"

        batch_size = self.params.get("batch_size")
        batch_size = int(batch_size) if batch_size is not None else "auto"

        explanation = self.explainer(
            X_np,
            max_evals=max_evals,
            batch_size=batch_size
        )

        values = np.asarray(explanation.values)

        values = self._select_values(
            shap_values=values,
            targets=targets
        )

        if self.params.get("abs", False):
            values = np.abs(values)

        return values