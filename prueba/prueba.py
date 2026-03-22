import argparse
from data_processing import preprocess_dataset
from models import usar_iforest, usar_ecod, usar_autoencoder, usar_hbos, usar_mcd, usar_vae
import pandas as pd
from xai import usar_shap_local, usar_lime
from config import DATASETS
from utils import QuantusWrapper
import numpy as np
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de XAI para PdM")
    parser.add_argument("-e", "--experiment", type=str, required=True,
                        choices=["shap", "lime"],
                        help="Tipo de experimento a ejecutar")
    
    args = parser.parse_args()
    experiment_type = args.experiment

    X_train, y_train, _, _, X_train_norm, _, anomalias_fraccion = preprocess_dataset('hydraulic', False)

    modelos = {
        # 'IForest': usar_iforest,
        'ECOD': usar_ecod,
        # 'AutoEncoder': usar_autoencoder,
        # 'HBOS': usar_hbos,
        # 'MCD': usar_mcd,
        # 'VAE': usar_vae,
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

            if experiment_type == "lime":
                explicaciones = usar_shap_local(model, model_name, 'hydraulic', X_train_norm, X_ev_norm, DATASETS['hydraulic']['observations'], False)

                evaluar_lime(model, X_train_norm, X_ev_norm, y_ev, explicaciones, 'hydraulic', model_name)



def evaluar_shap(model, X_test, y_test, explicaciones, dataset_name, model_name):
    wrapped_model = QuantusWrapper(model)

    from XAI_metrics.runner import run_all_metrics
    from XAI_metrics.base import MetricContext
    from XAI_metrics.reporting import save_metrics_report

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
    
    metric_results = run_all_metrics(
        ctx,
        selected_metrics=[
            "Complexity",
            "Sparseness",
            "Consistency",
            "FaithfulnessEstimate",
            "MonotonicityCorrelation",
            "Monotonicity",
            "SensitivityN",
            "Sufficiency",
            "Completeness",
            "NonSensitivity"
        ],
        config="XAI_metrics/config.yaml")
    print(metric_results)

    report_paths = save_metrics_report(
        metric_results=metric_results,
        output_dir=Path("results") / "metric_reports",
        report_name=f"{dataset_name}_{model_name}_metrics_report",
        observations=DATASETS["hydraulic"]["observations"],  # para detalle por observación
    )

    print("\nReportes guardados:")
    for fmt, path in report_paths.items():
        print(f"- {fmt}: {path}")


def evaluar_lime(model, X_train_bg, X_test, y_test, explicaciones, dataset_name, model_name):
    wrapped_model = QuantusWrapper(model)

    from XAI_metrics.runner import run_all_metrics
    from XAI_metrics.base import MetricContext
    from XAI_metrics.reporting import save_metrics_report

    def make_explain_func_lime(dataset_name: str, X_background, feature_names=None):
        from lime.lime_tabular import LimeTabularExplainer

        if isinstance(X_background, pd.DataFrame):
            cols = list(X_background.columns)
            X_bg_df = X_background.copy()
        else:
            X_bg_np = np.asarray(X_background)
            if feature_names is None:
                cols = [f"f{i}" for i in range(X_bg_np.shape[1])]
            else:
                cols = list(feature_names)
            X_bg_df = pd.DataFrame(X_bg_np, columns=cols)

        # Explainer una sola vez (más eficiente)
        lime_explainer = LimeTabularExplainer(
            training_data=X_bg_df.to_numpy(dtype=float, copy=True),
            feature_names=cols,
            random_state=42
        )

        def explain_func(model, inputs, targets=None, **kwargs):
            # Copia writable para evitar warning de quantus/torch
            X_np = (
                inputs.detach().cpu().numpy().copy()
                if isinstance(inputs, torch.Tensor)
                else np.array(inputs, dtype=float, copy=True)
            )
            if X_np.ndim == 1:
                X_np = X_np.reshape(1, -1)

            attributions = []
            for row in X_np:
                explanation = lime_explainer.explain_instance(
                    data_row=row.astype(float, copy=True),  # vector numpy, no Series
                    predict_fn=model.model.predict_proba,
                    num_features=len(cols),
                )

                pesos = np.zeros(len(cols), dtype=float)
                for feat, weight in explanation.as_list():
                    for i, col in enumerate(cols):
                        if str(col) in feat:
                            pesos[i] = float(weight)
                            break
                attributions.append(pesos)

            return np.asarray(attributions, dtype=float)

        return explain_func

    explain_func = make_explain_func_lime(
        dataset_name=dataset_name,
        X_background=X_train_bg,
        feature_names=getattr(X_test, "columns", None),
    )

    ctx = MetricContext(
        model=wrapped_model,
        X_test=X_test,
        y_test=y_test,
        observations=DATASETS["hydraulic"]["observations"],
        attributions=explicaciones,
        extras={"explain_func": explain_func},
    )

    metric_results = run_all_metrics(
        ctx,
        selected_metrics=[
            "Complexity",
            "Sparseness",
            "Consistency",
            "FaithfulnessEstimate",
            "MonotonicityCorrelation",
            "Monotonicity",
            "SensitivityN",
            "Sufficiency",
            "Completeness",
            "NonSensitivity"
        ],
        config="XAI_metrics/config.yaml"
    )
    print(metric_results)

    report_paths = save_metrics_report(
        metric_results=metric_results,
        output_dir=Path("results") / "metric_reports",
        report_name=f"{dataset_name}_{model_name}_metrics_report",
        observations=DATASETS["hydraulic"]["observations"],  # para detalle por observación
    )

    print("\nReportes guardados:")
    for fmt, path in report_paths.items():
        print(f"- {fmt}: {path}")


if __name__ == "__main__":
    main()