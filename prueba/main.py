import argparse
import os
import pandas as pd
import cloudpickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from xai_metrics.runner import run_evaluation

from xai_methods.break_down import usar_breakdown, evaluar_breakdown, make_breakdown_explain_func
from xai_methods.lime import usar_lime, evaluar_lime, make_lime_explain_func
from xai_methods.shap import usar_shap_local, evaluar_shap_local, make_shap_local_explain_func

from utils import load_model
from data_processing import preprocess_dataset
from config import BASE_DIR, DATASETS
from models import (
    usar_iforest,
    usar_ecod,
    usar_autoencoder,
    usar_hbos,
    usar_mcd,
    usar_vae
)


def guardar_atribuciones(explicaciones, observaciones, cols, dataset_name, model_name, method_name):

    output_dir = os.path.join(BASE_DIR, "prueba", "results", "attributions", dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    atribuciones_df = pd.DataFrame(
        explicaciones,
        index=observaciones,
        columns=cols
    )

    atribuciones_df.index.name = "observation"

    output_path = os.path.join(
        output_dir,
        f"{dataset_name}_{model_name}_{method_name}_attributions.csv"
    )

    atribuciones_df.to_csv(output_path, index=True)

    print(f"Atribuciones guardadas en: {output_path}")

def guardar_modelo(model, dataset_name, model_name, seed):

    class ExportedQuantusWrapper(nn.Module):
        def __init__(self, pyod_model):
            super().__init__()
            self.model = pyod_model

        def forward(self, inputs):
            probs = self.predict_proba(inputs)
            device = inputs.device if isinstance(inputs, torch.Tensor) else None
            return torch.as_tensor(probs, dtype=torch.float32, device=device)

        def _to_numpy(self, inputs):
            if isinstance(inputs, torch.Tensor):
                return inputs.detach().cpu().numpy().astype(np.float32)
            if hasattr(inputs, "to_numpy"):
                return inputs.to_numpy().astype(np.float32)
            return np.asarray(inputs, dtype=np.float32)

        def predict(self, inputs):
            return self.model.predict(self._to_numpy(inputs))

        def predict_proba(self, inputs):
            return self.model.predict_proba(self._to_numpy(inputs))

        def decision_function(self, inputs):
            return self.model.decision_function(self._to_numpy(inputs))

    output_dir = os.path.join(
        BASE_DIR,
        "prueba",
        "results",
        "models",
        dataset_name,
        model_name,
    )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{dataset_name}_{model_name}_seed_{seed}.pkl",
    )

    with open(output_path, "wb") as f:
        cloudpickle.dump(ExportedQuantusWrapper(model), f)

    print(f"Modelo guardado en: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de XAI para PdM")
    parser.add_argument("-e", "--experiment", type=str, required=True,
                        choices=["lime", "shap_local", "shap_global", "breakdown", "morris", "permutation", "eval"],
                        help="Tipo de experimento a ejecutar")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices={'hydraulic', 'AHU21'},
                        help="Nombre del dataset a usar")
    
    args = parser.parse_args()
    dataset_name = args.dataset
    experiment_type = args.experiment

    if dataset_name == "hydraulic":

        if experiment_type == "eval":

            X_train, y_train, _, _, X_train_norm, _, _ = preprocess_dataset(
                dataset_name,
                False
            )

            explain_funcs = {
                "lime": make_lime_explain_func(X_train_norm),
                "shap": make_shap_local_explain_func(X_train_norm),
                "breakdown": make_breakdown_explain_func(X_train_norm),
            }

            results = run_evaluation(
                config="xai_metrics/config.yaml",
                model_loader=load_model,
                explain_funcs=explain_funcs,
                report_output_dir=None # Comentar si se quiere guardar los reportes
            )

            # print(results)
            return 1

        X_train, y_train, _, _, X_train_norm, _, anomalias_fraccion = preprocess_dataset('hydraulic', False)


        output_dir = os.path.join(BASE_DIR, "prueba", "data", dataset_name)
        os.makedirs(output_dir, exist_ok=True)

        # X_train.to_csv(os.path.join(output_dir, "X_train.csv"), columns=list(X_train.columns), index=True)

        # X_train_norm.to_csv(os.path.join(output_dir, "X_train_norm.csv"), columns=list(X_train_norm.columns), index=True)

        # y_train.to_csv(os.path.join(output_dir, "y_train.csv"), header=["target"], index=True)

        modelos = {
            'IForest': usar_iforest,
            'ECOD': usar_ecod, # Este modelo es horrible
            'AutoEncoder': usar_autoencoder,
            'HBOS': usar_hbos,
            'MCD': usar_mcd,
            'VAE': usar_vae,
        }

        metrics_df = pd.DataFrame(columns=['Modelo', 'Semilla', 'Normalizado', 'Contaminacion', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'F1-score', 'Sensibilidad', 'Especificidad', 'Precisión', 'ROC-AUC', 'Tiempo (s)'])

        for model_name, model_function in modelos.items():
            print(f'\n********** {model_name} **********')

            metrics_df, model = model_function(X_train_norm, y_train, X_train_norm, y_train, metrics_df, True, anomalias_fraccion, 0)

            guardar_modelo(model, 'hydraulic', model_name, 0)

            if experiment_type == "lime":
                explicaciones = usar_lime(model, X_train_norm, X_train_norm, DATASETS['hydraulic']['observations'])

                print(explicaciones)
                # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train_norm.columns), 'hydraulic', model_name, 'lime')

                # evaluar_lime(model, X_train_norm, X_train_norm, y_train, explicaciones, 'hydraulic', model_name)

            if experiment_type == "shap_local":
                explicaciones = usar_shap_local(model, X_train_norm, X_train_norm, DATASETS['hydraulic']['observations'])

                print(explicaciones)
                # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train_norm.columns), 'hydraulic', model_name, 'shap_local')

                evaluar_shap_local(model, X_train_norm, X_train_norm, y_train, explicaciones, 'hydraulic', model_name)

            if experiment_type == "breakdown":
                explicaciones = usar_breakdown(
                    model,
                    X_train_norm,
                    X_train_norm,
                    DATASETS[dataset_name]["observations"]
                )

                print(explicaciones)
                # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train_norm.columns), 'hydraulic', model_name, 'breakdown')

                evaluar_breakdown(model, X_train_norm, X_train_norm, y_train, explicaciones, dataset_name, model_name)

    else:

        input_path = Path(BASE_DIR) / "prueba/data" / dataset_name
        X_train = pd.read_csv(input_path / "X_train.csv")
        y_train = pd.read_csv(input_path / "y_train.csv")
        X_val = pd.read_csv(input_path / "X_val.csv")
        y_val = pd.read_csv(input_path / "y_val.csv")
        X_test = pd.read_csv(input_path / "X_test.csv")
        y_test = pd.read_csv(input_path / "y_test.csv")

        output_dir = Path(BASE_DIR) / "prueba" / "results" / "datasets" / dataset_name
        output_dir.mkdir(exist_ok=True)

        modelos = {
            'IForest': usar_iforest,
            'ECOD': usar_ecod, # Este modelo es horrible
            'AutoEncoder': usar_autoencoder,
            'HBOS': usar_hbos,
            'MCD': usar_mcd,
            'VAE': usar_vae,
        }

        metrics_df = pd.DataFrame(columns=['Modelo', 'Semilla', 'Normalizado', 'Contaminacion', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'F1-score', 'Sensibilidad', 'Especificidad', 'Precisión', 'ROC-AUC', 'Tiempo (s)'])

        for model_name, model_function in modelos.items():
            print(f'\n********** {model_name} **********')

            y_total = np.concatenate([
                np.asarray(y_train).ravel(),
                np.asarray(y_val).ravel(),
                np.asarray(y_test).ravel(),
            ])
            anomalias_fraccion = np.mean(y_total == 1)

            metrics_df, model = model_function(X_train, y_train, X_val, y_val, X_test, y_test, metrics_df, True, anomalias_fraccion, 0)

            # guardar_modelo(model, 'hydraulic', model_name, 0)

            # if experiment_type == "lime":
            #     explicaciones = usar_lime(model, X_train, X_train, DATASETS['hydraulic']['observations'])

            #     print(explicaciones)
            #     # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train.columns), 'hydraulic', model_name, 'lime')

            #     # evaluar_lime(model, X_train, X_train, y_train, explicaciones, 'hydraulic', model_name)

            # if experiment_type == "shap_local":
            #     explicaciones = usar_shap_local(model, X_train, X_train, DATASETS['hydraulic']['observations'])

            #     print(explicaciones)
            #     # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train.columns), 'hydraulic', model_name, 'shap_local')

            #     evaluar_shap_local(model, X_train, X_train, y_train, explicaciones, 'hydraulic', model_name)

            # if experiment_type == "breakdown":
            #     explicaciones = usar_breakdown(
            #         model,
            #         X_train,
            #         X_train,
            #         DATASETS[dataset_name]["observations"]
            #     )

            #     print(explicaciones)
            #     # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train.columns), 'hydraulic', model_name, 'breakdown')

            #     evaluar_breakdown(model, X_train, X_train, y_train, explicaciones, dataset_name, model_name)

        



if __name__ == "__main__":
    main()
