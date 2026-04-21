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


def _get_reference_vector(X_train, strategy="median"):
    if isinstance(X_train, pd.DataFrame):
        X_np = X_train.to_numpy(dtype=float, copy=True)
    else:
        X_np = np.asarray(X_train, dtype=float)

    if strategy == "mean":
        ref = np.mean(X_np, axis=0)
    elif strategy == "median":
        ref = np.median(X_np, axis=0)
    elif strategy == "zero":
        ref = np.zeros(X_np.shape[1], dtype=float)
    else:
        raise ValueError("strategy debe ser 'mean', 'median' o 'zero'")

    return ref


def _get_model_scores(clf, X):
    """
    Devuelve un score continuo por muestra.
    Prioridad:
    1) decision_function
    2) predict_proba -> columna positiva / anomalía
    3) predict
    """
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    elif hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        proba = np.asarray(proba)

        if proba.ndim == 2 and proba.shape[1] >= 2:
            scores = proba[:, 1]
        else:
            scores = proba.ravel()
    elif hasattr(clf, "predict"):
        scores = clf.predict(X)
    else:
        raise AttributeError("El modelo no tiene decision_function, predict_proba ni predict.")

    return np.asarray(scores, dtype=float).ravel()


def usar_occlusion_local(
    clf,
    clf_name,
    dataset_name,
    X_train,
    X_test,
    observaciones_id,
    reference="median",
    groups=None,
    score_mode="difference",
    show_plot=False
):
    """
    Devuelve atribuciones locales por occlusion con shape (n_obs, n_features_o_grupos).

    Parámetros
    ----------
    clf : modelo entrenado
    X_train : DataFrame o ndarray
        Usado para construir el vector de referencia.
    X_test : DataFrame o ndarray
    observaciones_id : lista de ids/índices
    reference : {'median', 'mean', 'zero'}
    groups : dict o None
        Si None, occlusion por feature.
        Si dict, por grupos:
            {"grupo1": [0,1], "grupo2": [2,3]}
    score_mode : {'difference', 'relative'}
        difference: score_base - score_occ
        relative:   (score_base - score_occ) / (abs(score_base) + 1e-8)
    """
    ref_vec = _get_reference_vector(X_train, strategy=reference)

    if isinstance(X_test, pd.DataFrame):
        X_obs = X_test.loc[observaciones_id].copy()
        X_obs_np = X_obs.to_numpy(dtype=float, copy=True)
        feature_names = list(X_test.columns)
    else:
        X_test_np = np.asarray(X_test, dtype=float)
        X_obs_np = X_test_np[observaciones_id].copy()
        feature_names = [f"f{i}" for i in range(X_obs_np.shape[1])]

    n_obs, n_features = X_obs_np.shape

    if groups is None:
        groups = {feature_names[j]: [j] for j in range(n_features)}

    group_names = list(groups.keys())

    scores_base = _get_model_scores(clf, X_obs_np)
    attributions = np.zeros((n_obs, len(group_names)), dtype=float)

    for g_idx, g_name in enumerate(group_names):
        idxs = groups[g_name]

        X_occ = X_obs_np.copy()
        X_occ[:, idxs] = ref_vec[idxs]

        scores_occ = _get_model_scores(clf, X_occ)

        if score_mode == "difference":
            attributions[:, g_idx] = scores_base - scores_occ
        elif score_mode == "relative":
            attributions[:, g_idx] = (scores_base - scores_occ) / (np.abs(scores_base) + 1e-8)
        else:
            raise ValueError("score_mode debe ser 'difference' o 'relative'")

    return attributions