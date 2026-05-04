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
            X_test.loc[example].to_numpy(dtype=float, copy=True),
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


from interpret.blackbox import MorrisSensitivity
import numpy as np
import pandas as pd


def usar_morris_global(clf, clf_name, dataset_name, X_train):
    def predict_fn(X):
        X_np = X.to_numpy(dtype=float, copy=False) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return np.asarray(clf.decision_function(X_np), dtype=float).ravel()

    msa = MorrisSensitivity(predict_fn, X_train)
    explanation = msa.explain_global()
    data = explanation.data()

    feature_names = list(data["names"])
    scores = np.asarray(data["scores"], dtype=float).ravel()
    convergence_index = data.get("convergence_index", None)

    if len(feature_names) != len(scores):
        raise ValueError(
            f"Longitudes incompatibles: {len(feature_names)} nombres vs {len(scores)} scores"
        )

    scores_df = pd.DataFrame({
        "Feature": feature_names,
        "Score": scores,
        "Modelo": clf_name,
        "Dataset": dataset_name,
        "Metodo": "morris",
        "ConvergenceIndex": convergence_index,
    }).sort_values("Score", ascending=False, ignore_index=True)

    return scores_df, explanation


# import numpy as np
# import pandas as pd
# from alibi.explainers import ALE


# def usar_ale_global(
#     clf,
#     clf_name,
#     dataset_name,
#     X_train,
#     importances_df,
#     grid_points=20,
#     summary_mode="range"
# ):
#     if isinstance(X_train, pd.DataFrame):
#         feature_names = list(X_train.columns)
#         X_train_np = X_train.to_numpy(dtype=float, copy=True)
#     else:
#         X_train_np = np.asarray(X_train, dtype=float)
#         feature_names = [f"f{i}" for i in range(X_train_np.shape[1])]

#     def predict_fn(X):
#         X = np.asarray(X, dtype=float)

#         if hasattr(clf, "decision_function"):
#             y = clf.decision_function(X)
#         elif hasattr(clf, "predict_proba"):
#             proba = np.asarray(clf.predict_proba(X))
#             y = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
#         elif hasattr(clf, "predict"):
#             y = clf.predict(X)
#         else:
#             raise AttributeError("El modelo no tiene decision_function, predict_proba ni predict.")

#         return np.asarray(y, dtype=float).ravel()

#     explainer = ALE(
#         predictor=predict_fn,
#         feature_names=feature_names
#     )

#     explanation = explainer.explain(X_train_np, min_bin_points=4, grid_points=grid_points)
#     data = explanation.data
#     print("ALE keys:", data.keys())

#     # Normalmente data["ale_values"] trae una lista/estructura por feature
#     ale_values = data.get("ale_values", None)
#     if ale_values is None:
#         raise ValueError(f"No encuentro 'ale_values' en explanation.data: {data.keys()}")

#     scores = []
#     for vals in ale_values:
#         vals = np.asarray(vals, dtype=float).ravel()

#         if summary_mode == "range":
#             score = float(np.max(vals) - np.min(vals))
#         elif summary_mode == "var":
#             score = float(np.var(vals))
#         else:
#             raise ValueError("summary_mode debe ser 'range' o 'var'")

#         scores.append(score)

#     scores = np.asarray(scores, dtype=float)

#     model_feature_importance = pd.Series(scores, index=feature_names)
#     model_feature_ranking = model_feature_importance.rank(method="average", ascending=False).astype(int)
#     model_feature_ranking["Modelo"] = clf_name

#     importances_df = pd.concat(
#         [importances_df, pd.DataFrame([model_feature_ranking])],
#         ignore_index=True
#     )

#     return importances_df, explanation

# xai.py

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class PyODSklearnWrapper:
    def __init__(self, model, metric="f1"):
        self.model = model
        self.metric = metric

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        X_np = X.to_numpy(dtype=float, copy=False) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        return np.asarray(self.model.predict(X_np), dtype=int).ravel()

    def score(self, X, y):
        y_true = np.asarray(y).astype(int).ravel()
        y_pred = self.predict(X)

        if self.metric == "f1":
            return f1_score(y_true, y_pred)
        if self.metric == "accuracy":
            return accuracy_score(y_true, y_pred)
        if self.metric == "precision":
            return precision_score(y_true, y_pred, zero_division=0)
        if self.metric == "recall":
            return recall_score(y_true, y_pred, zero_division=0)

        raise ValueError("metric debe ser 'f1', 'accuracy', 'precision' o 'recall'")


def usar_permutation_sklearn(
    clf,
    clf_name,
    dataset_name,
    X_eval,
    y_eval,
    metric="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=1,
):
    wrapper = PyODSklearnWrapper(clf, metric=metric)

    if isinstance(X_eval, pd.DataFrame):
        feature_names = list(X_eval.columns)
        X_eval_np = X_eval.to_numpy(dtype=float, copy=True)
    else:
        X_eval_np = np.asarray(X_eval, dtype=float)
        feature_names = [f"f{i}" for i in range(X_eval_np.shape[1])]

    y_eval_np = np.asarray(y_eval).astype(int).ravel()

    result = permutation_importance(
        estimator=wrapper,
        X=X_eval_np,
        y=y_eval_np,
        scoring=None,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importances_mean = np.asarray(result.importances_mean, dtype=float)
    importances_std = np.asarray(result.importances_std, dtype=float)

    scores_df = pd.DataFrame({
        "Feature": feature_names,
        "Score": importances_mean,
        "Std": importances_std,
        "Modelo": clf_name,
        "Dataset": dataset_name,
        "Metric": metric,
        "Metodo": "permutation_sklearn",
    }).sort_values("Score", ascending=False, ignore_index=True)

    return scores_df, result