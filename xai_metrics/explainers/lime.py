# xai_metrics/explainers/lime.py
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from torch.nn import Module
from torch import Tensor

from xai_metrics.base import BaseExplainer, register_explainer, ExplainerContext

from typing import Any, Mapping

@register_explainer
class LIMEExplainer(BaseExplainer):
    NAME = "LIME"

    def __init__(
        self,
        context: ExplainerContext,
        params: Mapping[str, Any] | None = None
    ):
        super().__init__(context, params)

        self.cols = list(context.X_background.columns)

        self.explainer = LimeTabularExplainer(
            training_data=context.X_background.to_numpy(),
            mode=self.params.get("mode", "classification"),
            training_labels=None if context.y_background is None else context.y_background.to_numpy(),
            feature_names=self.cols,
            categorical_features=self.params.get("categorical_features"),
            categorical_names=self.params.get("categorical_names"),
            kernel_width=self.params.get("kernel_width"),
            kernel=self.params.get("kernel"),
            verbose=self.params.get("verbose", False),
            class_names=self.params.get("class_names"),
            feature_selection=self.params.get("feature_selection", "auto"),
            discretize_continuous=self.params.get("discretize_continuous", True),
            discretizer=self.params.get("discretizer", "quartile"),
            sample_around_instance=self.params.get("sample_around_instance", False),
            random_state=self.params.get("random_state", 42)
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

    
    def explain(
        self,
        model: Module,
        inputs: Any,
        targets: Any | None = None,
        **kwargs: Any
    ) -> np.ndarray:
        X_np = self._to_numpy(inputs)
        predict_fn = self._get_predict_fn(model)

        labels = self.params.get("labels", None)
        top_labels = self.params.get("top_labels", None)
        num_features = self.params.get("num_features", len(self.cols))
        num_samples = self.params.get("num_samples", 500)
        distance_metric = self.params.get("distance_metric", "euclidean")
        model_regressor = self.params.get("model_regressor", None)

        attributions = []
        
        for i, row in enumerate(X_np):
            if labels is None and top_labels is None:
                if targets is not None:
                    target = int(np.asarray(targets).reshape(-1)[i])
                    labels_to_explain = (target,)
                else:
                    labels_to_explain = (1,)
            else:
                labels_to_explain = labels

            explanation = self.explainer.explain_instance(
                data_row=row,
                predict_fn=predict_fn,
                labels=labels_to_explain,
                top_labels=top_labels,
                num_features=num_features,
                num_samples=num_samples,
                distance_metric=distance_metric,
                model_regressor=model_regressor
            )

            if top_labels is not None:
                label_to_use = explanation.top_labels[0]
            else:
                label_to_use = labels_to_explain[0]

            weights = np.zeros(len(self.cols), dtype=float)

            for feature_idx, weight in explanation.as_map()[label_to_use]:
                weights[int(feature_idx)] = float(weight)

            attributions.append(weights)

        return np.asarray(attributions)