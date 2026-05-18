import argparse
import pandas as pd

from data_processing import preprocess_dataset
from models import (
    usar_iforest,
    usar_ecod,
    usar_autoencoder,
    usar_hbos,
    usar_mcd,
    usar_vae
)
from xai_methods.lime import usar_lime, evaluar_lime
from xai_methods.shap import usar_shap_local, evaluar_shap_local
from xai_methods.break_down import usar_breakdown, evaluar_breakdown
from utils import DATASETS
from config import DATA_RAW_DIR

def main():
    parser = argparse.ArgumentParser(description="Ejecuta experimentos de XAI para PdM")
    parser.add_argument("-e", "--experiment", type=str, required=True,
                        choices=["lime", "shap_local", "shap_global", "breakdown", "morris", "permutation"],
                        help="Tipo de experimento a ejecutar")
    
    args = parser.parse_args()
    experiment_type = args.experiment

    X_train, y_train, _, _, X_train_norm, _, anomalias_fraccion = preprocess_dataset('hydraulic', False)

    X_train.to_csv(DATA_RAW_DIR)

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

        if experiment_type == "lime":
            explicaciones = usar_lime(model, X_train_norm, X_train_norm, DATASETS['hydraulic']['observations'])

            print(explicaciones)

            evaluar_lime(model, X_train_norm, X_train_norm, y_train, explicaciones, 'hydraulic', model_name)

        if experiment_type == "shap_local":
            explicaciones = usar_shap_local(model, X_train_norm, X_train_norm, DATASETS['hydraulic']['observations'])

            print(explicaciones)

            evaluar_shap_local(model, X_train_norm, X_train_norm, y_train, explicaciones, 'hydraulic', model_name)

        if experiment_type == "breakdown":
            explicaciones = usar_breakdown(
                model,
                X_train_norm,
                X_train_norm,
                DATASETS["hydraulic"]["observations"]
            )

            print(explicaciones)

            evaluar_breakdown(
                model,
                X_train_norm,
                X_train_norm,
                y_train,
                explicaciones,
                "hydraulic",
                model_name
            )



if __name__ == "__main__":
    main()