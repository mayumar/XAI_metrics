import torch.nn as nn
import torch
import numpy as np
import pandas as pd
from xai import usar_lime, usar_shap_local
from config import DATASETS

class QuantusWrapper(nn.Module):
    def __init__(self, pyod_model):
        super().__init__() # -> Inicializar un modelo base de PyTorch
        self.model = pyod_model

    # Método que PyTorch invoca cuando se hace wrapped_model(x) (es la interfaz estándar de nn.Module)
    # Debe aceptar tensores torch.Tensor y devolver otro tensor (o estructura compatible) con el resultado
    # x se supone que es un torch.Tensor
    # def forward(self, x):
    #     # convertir tensor a numpy
    #     x_np = x.detach().cpu().numpy() # x.detach() corta la relación con el grafo de cálculo (no se calcularán gradientes respecto a x)
    #     # Llama a decision_function del modelo de PyOD
    #     # Normalmente devuelve un array numpy con scores de anomalía (un número por cada fila de x_np)
    #     scores = self.model.decision_function(x_np)
    #     # convertir de vuelta a tensor
    #     scores = torch.tensor(scores, dtype=torch.float32)
    #     return scores

    def forward(self, x):
        # Asegurar formato float32
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.array(x, dtype=np.float32)
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_np)

        else:
            # obtener scores de anomalía
            scores = self.model.decision_function(x_np).astype(np.float32)

            # normalizar a [0, 1] si no lo están
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            # convertir a probabilidades 2D: [normal, anomalía]
            probs = np.stack([1 - scores, scores], axis=1).astype(np.float32)  # shape (N, 2)
            
        return torch.tensor(probs, dtype=torch.float32)

    def predict(self, x):
        # Mismo control de tipo para coherencia
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.array(x, dtype=np.float32)
            
        scores = self.model.decision_function(x_np).astype(np.float32)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        probs = np.stack([1 - scores, scores], axis=1).astype(np.float32)  # shape (N, 2)

        return probs

    def predict_proba(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy().astype(np.float32)
        else:
            x_np = np.array(x, dtype=np.float32)
        return torch.tensor(self.model.predict_proba(x_np), dtype=torch.float32)
    

def save_metric(name, results, dataset_name, metric_results):
    """
    Guarda los resultados de cada métrica en un formato tabular:
    - Si results es iterable con varios elementos → una columna por índice.
    - Si results es escalar o un único valor → una columna llamada 'media'.
    """
    row = {"Metric": name}
    
    # Si es un array/lista con varios elementos
    if isinstance(results, (list, tuple, np.ndarray, pd.Series)):
        if len(results) > 1:
            for i, val in enumerate(results):
                row[DATASETS[dataset_name]['observations'][i]] = val
        elif len(results) == 1:
            row["media"] = results[0]
        else:
            row["media"] = np.nan
    # Si es un dict (algunas métricas devuelven un dict con valores agregados)
    elif isinstance(results, dict):
        row.update(results)
    # Si es un único valor escalar
    else:
        row["media"] = results
    
    metric_results.append(row)

    return metric_results


def make_explain_func(X_test, observations, xai_method, dataset_name):
    def explain_func(model, inputs, targets=None, **kwargs):
        """
        Función wrapper que adapta LIME para Quantus.
        - model: el modelo (wrapped_model en tu caso)
        - inputs: batch de instancias a explicar (numpy o tensor)
        - targets: no siempre necesario para LIME
        """
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        if not isinstance(inputs, pd.DataFrame):
            inputs = pd.DataFrame(inputs, columns=X_test.columns, index=observations)

        if xai_method == 'LIME':
            # Generar explicaciones con usar_lime
            explicaciones = usar_lime(
                clf=model.model,  # pasar el modelo original de pyod
                clf_name=None,
                dataset_name=dataset_name,
                X_train=X_test,
                X_test=inputs,
                observaciones=observations
            )
        elif xai_method == "SHAP":
            explicaciones = usar_shap_local(
                clf=model.model,
                clf_name=None,
                dataset_name=dataset_name,
                X_train=X_test,
                X_test=inputs,
                observaciones_id=observations,
                show_plot=False
            )

        return explicaciones  # Quantus espera un array-like
    
    return explain_func