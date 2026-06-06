import argparse
import os
import pandas as pd

def guardar_atribuciones(explicaciones, observaciones, cols, dataset_name, model_name, method_name):
    from config import BASE_DIR

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
    import os
    import cloudpickle
    import torch
    import torch.nn as nn
    import numpy as np

    from config import BASE_DIR

    class ExportedQuantusWrapper(nn.Module):
        def __init__(self, pyod_model):
            super().__init__()
            self.model = pyod_model

        def forward(self, inputs):
            return self.predict_proba(inputs)

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
    
    args = parser.parse_args()
    experiment_type = args.experiment

    if experiment_type == "eval":
        from xai_metrics.runner import run_evaluation
        from utils import load_model
        from config import DATASETS
        from data_processing import preprocess_dataset

        from xai_methods.lime import make_lime_explain_func
        from xai_methods.shap import make_shap_local_explain_func
        from xai_methods.break_down import make_breakdown_explain_func

        X_train, y_train, _, _, X_train_norm, _, _ = preprocess_dataset(
            "hydraulic",
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

    from config import BASE_DIR, DATASETS
    from data_processing import preprocess_dataset
    from models import (
        usar_iforest,
        usar_ecod,
        usar_autoencoder,
        usar_hbos,
        usar_mcd,
        usar_vae
    )
    from xai_methods.break_down import usar_breakdown, evaluar_breakdown
    from xai_methods.lime import usar_lime, evaluar_lime
    from xai_methods.shap import usar_shap_local, evaluar_shap_local

    X_train, y_train, _, _, X_train_norm, _, anomalias_fraccion = preprocess_dataset('hydraulic', False)


    output_dir = os.path.join(BASE_DIR, "prueba", "data", "hydraulic")
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
                DATASETS["hydraulic"]["observations"]
            )

            print(explicaciones)
            # guardar_atribuciones(explicaciones, DATASETS['hydraulic']['observations'], list(X_train_norm.columns), 'hydraulic', model_name, 'breakdown')

            evaluar_breakdown(model, X_train_norm, X_train_norm, y_train, explicaciones, "hydraulic", model_name)



if __name__ == "__main__":
    main()
