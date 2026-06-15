import argparse
import os
import pandas as pd
import cloudpickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle
import joblib

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
from optimizacion_hiperparametros import optimizar_modelo, entrenar_modelo_optimizado


def seleccionar_observaciones(
    model,
    X_test,
    y_test,
    n=10,
    random_state=0,
):
    y_true = np.asarray(y_test).ravel()
    y_pred = model.predict(X_test)

    grupos = [
        np.where((y_true == 1) & (y_pred == 1))[0],  # TP
        np.where((y_true == 1) & (y_pred == 0))[0],  # FN
        np.where((y_true == 0) & (y_pred == 1))[0],  # FP
        np.where((y_true == 0) & (y_pred == 0))[0],  # TN
    ]

    rng = np.random.default_rng(random_state)
    seleccion = []

    for grupo in grupos:
        if len(grupo) > 0:
            seleccion.extend(
                rng.choice(
                    grupo,
                    size=min(n, len(grupo)),
                    replace=False,
                )
            )

    return sorted(seleccion)


def guardar_atribuciones(explicaciones, observaciones, cols, dataset_name, model_name, method_name):

    output_dir = os.path.join(BASE_DIR, "prueba", "results", "attributions", dataset_name, model_name, method_name)
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
                        choices=["lime", "shap_local", "shap_global", "breakdown", "eval", "optuna", "train"],
                        help="Tipo de experimento a ejecutar")
    parser.add_argument("-d", "--dataset", type=str, required=True, choices={'hydraulic', 'AHU21', 'ARAMIS20', 'PHM14', 'HSG18'},
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

        # X_train, y_train, _, _, X_train_norm, _, anomalias_fraccion = preprocess_dataset('hydraulic', False)


        # output_dir = os.path.join(BASE_DIR, "prueba", "data", dataset_name)
        # os.makedirs(output_dir, exist_ok=True)

        # # X_train.to_csv(os.path.join(output_dir, "X_train.csv"), columns=list(X_train.columns), index=True)

        # # X_train_norm.to_csv(os.path.join(output_dir, "X_train_norm.csv"), columns=list(X_train_norm.columns), index=True)

        # # y_train.to_csv(os.path.join(output_dir, "y_train.csv"), header=["target"], index=True)

        # modelos = {
        #     'IForest': usar_iforest,
        #     'ECOD': usar_ecod, # Este modelo es horrible
        #     'AutoEncoder': usar_autoencoder,
        #     'HBOS': usar_hbos,
        #     'MCD': usar_mcd,
        #     'VAE': usar_vae,
        # }

        # metrics_df = pd.DataFrame(columns=['Modelo', 'Semilla', 'Normalizado', 'Contaminacion', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'F1-score', 'Sensibilidad', 'Especificidad', 'Precisión', 'ROC-AUC', 'Tiempo (s)'])

        # for model_name, model_function in modelos.items():
        #     print(f'\n********** {model_name} **********')

        #     metrics_df, model = model_function(X_train_norm, y_train, X_train_norm, y_train, metrics_df, True, anomalias_fraccion, 0)

        #     guardar_modelo(model, 'hydraulic', model_name, 0)

        #     if experiment_type == "lime":
        #         explicaciones = usar_lime(model, X_train_norm, X_train_norm, DATASETS['hydraulic']['observations'])

        #         print(explicaciones)
        #         # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train_norm.columns), 'hydraulic', model_name, 'lime')

        #         # evaluar_lime(model, X_train_norm, X_train_norm, y_train, explicaciones, 'hydraulic', model_name)

        #     if experiment_type == "shap_local":
        #         explicaciones = usar_shap_local(model, X_train_norm, X_train_norm, DATASETS['hydraulic']['observations'])

        #         print(explicaciones)
        #         # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train_norm.columns), 'hydraulic', model_name, 'shap_local')

        #         evaluar_shap_local(model, X_train_norm, X_train_norm, y_train, explicaciones, 'hydraulic', model_name)

        #     if experiment_type == "breakdown":
        #         explicaciones = usar_breakdown(
        #             model,
        #             X_train_norm,
        #             X_train_norm,
        #             DATASETS[dataset_name]["observations"]
        #         )

        #         print(explicaciones)
        #         # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train_norm.columns), 'hydraulic', model_name, 'breakdown')

        #         evaluar_breakdown(model, X_train_norm, X_train_norm, y_train, explicaciones, dataset_name, model_name)

    else:

        input_path = Path(BASE_DIR) / "prueba/data" / dataset_name
        X_train = pd.read_csv(input_path / "X_train.csv", index_col=0)
        y_train = pd.read_csv(input_path / "y_train.csv", index_col=0)
        X_val = pd.read_csv(input_path / "X_val.csv", index_col=0)
        y_val = pd.read_csv(input_path / "y_val.csv", index_col=0)
        X_test = pd.read_csv(input_path / "X_test.csv", index_col=0)
        y_test = pd.read_csv(input_path / "y_test.csv", index_col=0)

        output_dir = Path(BASE_DIR) / "prueba" / "results" / "datasets" / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_type == "optuna" or experiment_type == "train":
            metrics_path = output_dir / f"{dataset_name}_model_metrics.csv"

            modelos_optimizados = [
                "IForest",
                "ECOD",
                "HBOS",
                "MCD",
                "AutoEncoder",
                "VAE",
            ]

            metrics_df = pd.DataFrame(columns=['Modelo', 'Semilla', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'F1-score', 'Sensibilidad', 'Especificidad', 'Precisión', 'ROC-AUC', 'Params'])

            for model_name in modelos_optimizados:
                print(f'\n********** {model_name} **********')

                if experiment_type == "optuna":

                    optimizar_modelo(
                        dataset_name=dataset_name,
                        nombre_modelo=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        random_state=0,
                        n_trials=30
                    )

                elif experiment_type == "train":

                    metrics_df, model = entrenar_modelo_optimizado(
                        dataset=dataset_name,
                        nombre_modelo=model_name,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        X_test=X_test,
                        y_test=y_test,
                        metricas=metrics_df,
                        random_state=0
                    )

                    guardar_modelo(
                        model=model,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        seed=0
                    )

                    metrics_df.to_csv(metrics_path, index=False)

                    print(f"Metricas guardades en: {metrics_path}")

            print("\nResultados:")
            print(metrics_df.to_string(index=False))
        
        else:

            if experiment_type == "eval":
                # explain_funcs = {
                #     "lime": make_lime_explain_func(X_train),
                #     "shap": make_shap_local_explain_func(X_train),
                #     "breakdown": make_breakdown_explain_func(X_train),
                # }

                input_path_aramis = Path(BASE_DIR) / "prueba/data" / "ARAMIS20"
                input_path_ahu = Path(BASE_DIR) / "prueba/data" / "AHU21"
                X_train_aramis = pd.read_csv(input_path_aramis / "X_train.csv", index_col=0)
                X_train_ahu = pd.read_csv(input_path_ahu / "X_train.csv", index_col=0)

                explain_funcs = {
                    "ARAMIS20": {
                        "lime": make_lime_explain_func(X_train_aramis),
                        "shap": make_shap_local_explain_func(X_train_aramis),
                    },
                    "AHU21": {
                        "lime": make_lime_explain_func(X_train_ahu),
                        "shap": make_shap_local_explain_func(X_train_ahu),
                    },
                }

                results = run_evaluation(
                    config="xai_metrics/config.yaml",
                    model_loader=load_model,
                    explain_funcs=explain_funcs,
                    # report_output_dir=None # Comentar si se quiere guardar los reportes
                )

                print(results)
                return 1

            models_path = Path(BASE_DIR) / "prueba" / "results" / "models" / dataset_name

            for model_dir in models_path.iterdir():
                if model_dir.is_dir():
                    for model_path in model_dir.iterdir():
                        model_name = model_path.parent.name
                        model = load_model(model_path)
                        posiciones = seleccionar_observaciones(
                            model=model,
                            X_test=X_test,
                            y_test=y_test,
                            n=10,
                            random_state=0,
                        )

                        X_explicar = X_test.iloc[posiciones]
                        observaciones = X_test.index[posiciones].tolist()

                        if experiment_type == "lime":
                            explicaciones = usar_lime(model, X_train, X_explicar, observaciones)

                            print(explicaciones)
                            guardar_atribuciones(explicaciones, observaciones, X_train.columns, dataset_name, model_name, "lime")

                        if experiment_type == "shap_local":
                            explicaciones = usar_shap_local(model, X_train, X_explicar, observaciones)

                            print(explicaciones)
                            guardar_atribuciones(explicaciones, observaciones, X_train.columns, dataset_name, model_name, "shap_local")

            # metrics_df, model = model_function(X_train, y_train, X_val, y_val, X_test, y_test, metrics_df, True, 0)

            # else:

            #     metrics_df, model = model_function(X_train, y_train, X_val, y_val, X_test, y_test, metrics_df, True, anomalias_fraccion, 0)

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
