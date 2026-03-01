import argparse
from data_processing import preprocess_dataset
from models import usar_iforest, usar_ecod, usar_autoencoder, usar_hbos, usar_mcd, usar_vae
import pandas as pd
from xai import usar_shap_local
from config import DATASETS
from utils import QuantusWrapper
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de XAI para PdM")
    parser.add_argument("-e", "--experiment", type=str, required=True,
                        choices=["shap", "lime"],
                        help="Tipo de experimento a ejecutar")
    
    args = parser.parse_args()
    experiment_type = args.experiment

    X_train, y_train, _, _, X_train_norm, _, anomalias_fraccion = preprocess_dataset('hydraulic', False)

    modelos = {
        'IForest': usar_iforest,
        'ECOD': usar_ecod,
        'AutoEncoder': usar_autoencoder,
        'HBOS': usar_hbos,
        'MCD': usar_mcd,
        'VAE': usar_vae,
    }

    importances = pd.DataFrame()
    n_seeds = 1

    metrics_df = pd.DataFrame(columns=['Modelo', 'Semilla', 'Normalizado', 'Contaminacion', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'F1-score', 'Sensibilidad', 'Especificidad', 'Precisión', 'ROC-AUC', 'Tiempo (s)'])

    for model_name, model_function in modelos.items():
        print(f'\n********** {model_name} **********')
        for seed in range(n_seeds):
            print(f'\nSemilla: {seed}')

            X_ev = X_train.copy()
            X_ev_norm = X_train_norm.copy()
            y_ev = y_train.copy()

            metrics_df, model = model_function(X_train_norm, y_train, X_ev_norm, y_ev, metrics_df, True, anomalias_fraccion, seed)

            if experiment_type == "shap":
                explicaciones = usar_shap_local(model, model_name, 'hydraulic', X_train_norm, X_ev_norm, DATASETS['hydraulic']['observations'], False)

                evaluar_shap(model, X_ev_norm, y_ev, explicaciones, 'hydraulic', model_name)



def evaluar_shap(model, X_test, y_test, explicaciones, dataset_name, model_name):
    wrapped_model = QuantusWrapper(model)

    from XAI_metrics.runner import run_all_metrics
    from XAI_metrics.base import MetricContext

    def make_explain_func_shap(dataset_name: str, X_background, observations, feature_names=None):
        # Background + nombres de columnas
        if isinstance(X_background, pd.DataFrame):
            cols = list(X_background.columns)
            X_bg_df = X_background
        else:
            X_bg_np = np.asarray(X_background)
            if feature_names is None:
                cols = [f"f{i}" for i in range(X_bg_np.shape[1])]
            else:
                cols = list(feature_names)
            X_bg_df = pd.DataFrame(X_bg_np, columns=cols)

        def explain_func(model, inputs, targets=None, **kwargs):
            X_np = inputs.detach().cpu().numpy() if isinstance(inputs, torch.Tensor) else np.asarray(inputs)

            # índice 0..n-1 para que .loc funcione con ids locales
            X_batch = pd.DataFrame(X_np, columns=cols, index=pd.RangeIndex(start=0, stop=len(X_np)))

            local_ids = list(X_batch.index)  # [0,1,2,...]

            shap_vals = usar_shap_local(
                clf=model.model,
                clf_name=None,
                dataset_name=dataset_name,
                X_train=X_bg_df,
                X_test=X_batch,
                observaciones_id=local_ids,
                show_plot=False
            )
            return np.asarray(shap_vals)

        return explain_func
    
    explain_func = make_explain_func_shap(
        dataset_name="hydraulic",
        X_background=X_test,                         # si es DataFrame, perfecto
        observations=explicaciones,
        feature_names=getattr(X_test, "columns", None)
    )

    ctx = MetricContext(
        model=wrapped_model,
        X_test=X_test,
        y_test=y_test,
        observations=DATASETS['hydraulic']['observations'],
        attributions=explicaciones,
        extras={"explain_func": explain_func}
        )
    
    metric_results = run_all_metrics(ctx)
    print(metric_results)



if __name__ == "__main__":
    main()