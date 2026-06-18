# xai_metrics/explainers/breakdown.py
import dalex as dx
from torch.nn import Module
import numpy as np
import pandas as pd
from torch import Tensor

from xai_metrics.base import register_explainer, BaseExplainer
from xai_metrics.base.base_explainer import ExplainerContext

from typing import Any, Mapping

@register_explainer
class BreakDownExplainer(BaseExplainer):
    NAME = "BreakDown"

    def __init__(
        self,
        context: ExplainerContext,
        params: Mapping[str, Any] | None = None
    ):
        super().__init__(context, params)

        if context.model is None:
            raise ValueError("BreakDownExplainer requires context.model.")
        
        self.cols = list(context.X_background.columns)
        
        self.explainer = dx.Explainer(
            model=context.model,
            data=context.X_background,
            y=context.y_background,
            predict_function=self._get_predict_fn(context.model),
            label=self.params.get("label", type(context.model).__name__),
            verbose=self.params.get("verbose", False),
            precalculate=self.params.get("precalculate", True),
            model_type=self.params.get("mode", "classification")
        )


    def _to_dataframe(self, inputs: Any) -> pd.DataFrame:
        if isinstance(inputs, pd.DataFrame):
            return inputs.loc[:, self.cols]

        if isinstance(inputs, Tensor):
            values = inputs.detach().cpu().numpy()
        else:
            values = np.asarray(inputs)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        return pd.DataFrame(values, columns=self.cols)


    def _get_predict_fn(self, model: Module, predict_fn=None):
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
    

    def explain(
        self,
        model: Module,
        inputs: Any,
        targets: Any | None = None,
        **kwargs: Any
    ) -> np.ndarray[tuple[Any, ...], np.dtype[Any]]:
        X_df = self._to_dataframe(inputs)

        attributions = []

        for _, row in X_df.iterrows():
            explanation = self.explainer.predict_parts(
                new_observation=row.to_frame().T,
                type="break_down",
                order=self.params.get("order"),
                N=self.params.get("n_samples"),
                keep_distributions=self.params.get("keep_distributions", False),
                random_state=self.params.get("random_state", 42)
            )

            attributions.append(self._result_to_weights(explanation.result))

        values = np.asarray(attributions)

        if self.params.get("abs", False):
            values = np.abs(values)

        return values