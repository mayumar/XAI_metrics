# src/xai_utils.py

import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from config import SHAP_DIR, LIME_DIR, PDP_DIR, DATASETS

def usar_shap_global(clf, clf_name, dataset_name, X_train, X_test, importances_df, show_plot=True):
    # Crear un SHAP explainer
    explainer = shap.Explainer(clf.predict, X_train)

    # Explicar los valores
    shap_values = explainer(X_test)

    # Generar los reportes
    shap.plots.beeswarm(shap_values, show=show_plot)

    if not show_plot:
        fig = plt.gcf()
        fig.savefig(os.path.join(os.path.join(SHAP_DIR, dataset_name), f"shap_{clf_name}.png"), bbox_inches='tight', dpi=300)
        plt.close(fig)

    # Extraemos los valores shap
    shap_array = shap_values.values  # <- Accedemos a los valores puros

    # Calculamos la importancia media usando valores absolutos
    shap_importance = np.abs(shap_array).mean(axis=0)

    # Creamos una fila para este modelo
    model_feature_importance = pd.Series(dict(zip(X_train.columns, shap_importance)))

    # Obtenemos el ranking (1 = más importante)
    model_feature_ranking = model_feature_importance.rank(method='average', ascending=False).astype(int)

    # Añadimos el modelo
    model_feature_ranking['Modelo'] = clf_name

    # Lo añadimos al DataFrame
    importances_df = pd.concat([importances_df, pd.DataFrame([model_feature_ranking])], ignore_index=True)

    return importances_df

def usar_shap_local(clf, clf_name, dataset_name, X_train, X_test, observaciones_id, show_plot=False):
    # Crear un SHAP explainer
    explainer = shap.Explainer(clf.predict, X_train)

    print(f"Explicando: {observaciones_id}")
    observaciones = X_test.loc[observaciones_id]
    
    # Explicar los valores
    shap_values = explainer(observaciones)

    for value in shap_values:
        # Generar los reportes
        shap.plots.waterfall(value, show=show_plot)

        if show_plot:
            fig = plt.gcf()
            fig.savefig(os.path.join(os.path.join(SHAP_DIR, dataset_name), f"shap_{clf_name}.png"), bbox_inches='tight', dpi=300)
            plt.close(fig)

    return np.abs(shap_values.values)


def usar_lime(clf, clf_name, dataset_name, X_train, X_test, observaciones, show_plot=False):
    lime_explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        random_state=42
    )

    explicaciones_lista = []  # primero en lista

    for example in observaciones:
        print(f"Explicando instancia: {example}")
        explanation = lime_explainer.explain_instance(
            X_test.loc[example],
            clf.predict_proba,
            num_features=len(X_train.columns)
        )

        if show_plot:
            # Guardar figura
            fig = explanation.as_pyplot_figure()
            fig.savefig(os.path.join(
                LIME_DIR,
                dataset_name,
                f"lime_explanation_{str(example).replace(' ', '_').replace(':', '-').replace('/', '-')}_{clf_name}.png"
            ), bbox_inches='tight')

        # Extraer solo los pesos en el orden de las features
        pesos = np.zeros(len(X_train.columns))
        for feat, weight in explanation.as_list():
            # feat es un string con la condición, hay que mapearlo al índice
            for i, col in enumerate(X_train.columns):
                if col in feat:  # detección simple (puedes refinar esto)
                    pesos[i] = weight
        explicaciones_lista.append(pesos)

    # Convertir a ndarray al final
    explicaciones = np.vstack(explicaciones_lista)

    return explicaciones
