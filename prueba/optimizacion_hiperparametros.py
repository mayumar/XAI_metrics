from pathlib import Path
import optuna
import numpy as np
import pandas as pd
import json
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from pyod.models.iforest import IForest
from pyod.models.cblof import CBLOF
from pyod.models.ecod import ECOD
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE

# Suponiendo que este archivo está dentro de src/.
PROJECT_DIR = Path(__file__).resolve().parents[1]

OPTUNA_DIR = PROJECT_DIR / "optuna_results"
BEST_PARAMS_DIR = OPTUNA_DIR / "best_params"
OPTUNA_DATABASE = OPTUNA_DIR / "studies.db"

OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
BEST_PARAMS_DIR.mkdir(parents=True, exist_ok=True)


def _as_numpy(X):
    if hasattr(X, "to_numpy"):
        return X.to_numpy()
    return np.asarray(X)


def _crear_modelo_optuna(trial: optuna.Trial, nombre_modelo: str, random_state: int):
    nombre_modelo = nombre_modelo.lower()

    contaminacion = trial.suggest_categorical(
        "contamination",
        [0.001, 0.005, 0.01, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20]
    )

    if nombre_modelo == "iforest":
        return IForest(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_samples=trial.suggest_categorical("max_samples", ["auto", 0.5, 0.75, 1.0]), # type: ignore
            contamination=contaminacion,
            max_features=trial.suggest_categorical("max_features", [0.5, 0.75, 1.0]),
            bootstrap=trial.suggest_categorical("bootstrap", [False, True]),
            n_jobs=-1,
            random_state=random_state
        )
    
    if nombre_modelo == "cblof":
        return CBLOF(
            n_clusters=trial.suggest_int("n_clusters", 3, 15),
            contamination=contaminacion,
            alpha=trial.suggest_float("alpha", 0.6, 0.95),
            beta=trial.suggest_int("beta", 2, 10),
            use_weights=trial.suggest_categorical("use_weights", [False, True]),
            n_jobs=-1,
            random_state=random_state
        )
    
    if nombre_modelo == "ecod":
        return ECOD(
            contamination=contaminacion,
            n_jobs=1
        )
    
    if nombre_modelo == "hbos":
        return HBOS(
            n_bins=trial.suggest_categorical("n_bins", ["auto", 5, 10, 20, 30, 50]), # type: ignore
            alpha=trial.suggest_float("alpha", 0.01, 0.5, log=True),
            tol=trial.suggest_float("tol", 0.1, 0.9),
            contamination=contaminacion
        )
    
    if nombre_modelo == "mcd":
        return MCD(
            contamination=contaminacion,
            assume_centered=trial.suggest_categorical("assume_centered", [False, True]),
            support_fraction=trial.suggest_categorical("support_fraction", [None, 0.6, 0.75, 0.9, 1.0]),#1.0]),#
            random_state=random_state
        )
    
    if nombre_modelo == "autoencoder":
        arquitectura = trial.suggest_categorical(
            "arquitectura",
            [
                "pequena",
                "media",
                "grande",
            ],
        )

        arquitecturas = {
            "pequena": [32, 16],
            "media": [64, 32],
            "grande": [128, 64, 32],
        }

        return AutoEncoder(
            contamination=contaminacion,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            epoch_num=trial.suggest_categorical("epoch_num", [10, 20, 30, 50]),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            hidden_neuron_list=arquitecturas[arquitectura],
            batch_norm=trial.suggest_categorical("batch_norm", [False, True]),
            dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1),
            verbose=0,
            random_state=random_state
        )
    
    if nombre_modelo == "vae":
        arquitectura = trial.suggest_categorical(
            "arquitectura",
            [
                "pequena",
                "media",
                "grande",
            ],
        )

        arquitecturas = {
            "pequena": {
                "encoder": [32, 16],
                "decoder": [16, 32],
            },
            "media": {
                "encoder": [64, 32],
                "decoder": [32, 64],
            },
            "grande": {
                "encoder": [128, 64, 32],
                "decoder": [32, 64, 128],
            },
        }

        return VAE(
            contamination=contaminacion,
            lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            epoch_num=trial.suggest_categorical("epoch_num", [10, 20, 30, 50]),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            encoder_neuron_list=arquitecturas[arquitectura]['encoder'],
            decoder_neuron_list=arquitecturas[arquitectura]['decoder'],
            latent_dim=trial.suggest_categorical("latent_dim", [2, 4, 8, 16]),
            batch_norm=trial.suggest_categorical("batch_norm", [False, True]),
            dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1),
            verbose=0,
            random_state=random_state
        )

    raise ValueError(
        f"Modelo no reconocido: {nombre_modelo}"
    )


def guardar_mejores_parametros(
    study,
    dataset,
    nombre_modelo,
):
    nombre_modelo = nombre_modelo.lower()

    ruta = BEST_PARAMS_DIR / f"{dataset}_{nombre_modelo}.json"

    datos = {
        "best_f1_validation": study.best_value,
        "params": study.best_params,
    }

    with ruta.open("w", encoding="utf-8") as archivo:
        json.dump(
            datos,
            archivo,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Mejores parámetros guardados en: {ruta}")


def crear_callback_guardado(dataset, nombre_modelo):
    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if trial.number == study.best_trial.number:
            guardar_mejores_parametros(
                study=study,
                dataset=dataset,
                nombre_modelo=nombre_modelo,
            )

    return callback


def cargar_mejores_parametros(
    dataset,
    nombre_modelo,
):
    nombre_modelo = nombre_modelo.lower()

    ruta = BEST_PARAMS_DIR / f"{dataset}_{nombre_modelo}.json"

    if not ruta.exists():
        raise FileNotFoundError(
            f"No existen parámetros guardados para "
            f"{nombre_modelo} en {dataset}: {ruta}"
        )

    with ruta.open("r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    return datos["params"]


def crear_modelo_con_mejores_parametros(
    dataset,
    nombre_modelo,
    random_state=0,
):
    parametros = cargar_mejores_parametros(
        dataset=dataset,
        nombre_modelo=nombre_modelo,
    )

    fixed_trial = optuna.trial.FixedTrial(parametros)

    modelo = _crear_modelo_optuna(
        trial=fixed_trial,
        nombre_modelo=nombre_modelo,
        random_state=random_state,
    )

    return modelo, parametros


def entrenar_modelo_optimizado(
    dataset,
    nombre_modelo,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    metricas,
    random_state=0,
):
    X_train = _as_numpy(X_train)
    X_val = _as_numpy(X_val)
    X_test = _as_numpy(X_test)

    y_train = np.asarray(y_train).ravel()
    y_val = np.asarray(y_val).ravel()
    y_test = np.asarray(y_test).ravel()

    # Cargar mejores parámetros y reconstruir el modelo.
    modelo, parametros = crear_modelo_con_mejores_parametros(
        dataset=dataset,
        nombre_modelo=nombre_modelo,
        random_state=random_state,
    )

    # Unir train y validación para el entrenamiento final.
    X_train_val = np.concatenate(
        [X_train, X_val],
        axis=0,
    )

    y_train_val = np.concatenate(
        [y_train, y_val],
        axis=0,
    )

    # AutoEncoder y VAE se entrenan solo con muestras normales.
    if nombre_modelo.lower() in {
        "autoencoder",
        "vae",
    }:
        X_fit = X_train_val[y_train_val == 0]
    else:
        X_fit = X_train_val

    print(
        f"Entrenando {nombre_modelo} con los mejores "
        f"hiperparámetros: {parametros}"
    )

    modelo.fit(X_fit)

    # Predicciones finales sobre test.
    y_pred = modelo.predict(X_test)

    # Puntuaciones continuas para calcular ROC-AUC.
    y_score = modelo.decision_function(X_test)

    # Matriz de confusión siempre en el orden 0, 1.
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    sensibilidad = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    especificidad = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    # ROC-AUC no está definido si test solo tiene una clase.
    if np.unique(y_test).size == 2:
        roc_auc = roc_auc_score(
            y_test,
            y_score,
        )
    else:
        roc_auc = np.nan

    nueva_fila = {
        "Modelo": nombre_modelo,
        "Semilla": random_state,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": accuracy,
        "F1-score": f1,
        "Sensibilidad": sensibilidad,
        "Especificidad": especificidad,
        "Precisión": precision,
        "ROC-AUC": roc_auc,
        # JSON permite guardar el diccionario completo en una celda del CSV.
        "Params": json.dumps(
            parametros,
            ensure_ascii=False,
            sort_keys=True,
        ),
    }

    nueva_fila_df = pd.DataFrame([nueva_fila])

    metricas = pd.concat(
        [metricas, nueva_fila_df],
        ignore_index=True,
    )

    print(f"Matriz de confusión de {nombre_modelo}:")
    print(cm)

    print(
        f"F1 test: {f1:.6f} | "
        f"ROC-AUC: {roc_auc:.6f}"
    )

    return metricas, modelo


def crear_estudio(
    dataset,
    nombre_modelo,
):
    nombre_estudio = f"{dataset}_{nombre_modelo}"

    storage = f"sqlite:///{OPTUNA_DATABASE.resolve()}"

    # estudios = optuna.study.get_all_study_summaries(
    #     storage=storage
    # )

    # nombres_estudios = {
    #     estudio.study_name
    #     for estudio in estudios
    # }

    # if nombre_estudio in nombres_estudios:
    #     optuna.delete_study(
    #         study_name=nombre_estudio,
    #         storage=storage,
    #     )

    #     print(
    #         f"Estudio anterior eliminado: "
    #         f"{nombre_estudio}"
    #     )

    return optuna.create_study(
        study_name=nombre_estudio,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
    )


def optimizar_modelo(
    dataset_name,
    nombre_modelo,
    X_train,
    y_train,
    X_val,
    y_val,
    random_state = 0,
    n_trials = 30
):
    nombre_modelo = nombre_modelo.lower()

    modelos_validos = {
        "cblof",
        "iforest",
        "ecod",
        "autoencoder",
        "hbos",
        "mcd",
        "vae"
    }

    if nombre_modelo not in modelos_validos:
        raise ValueError(
            f"Modelo no reconocido: {nombre_modelo}. "
            f"Opciones válidas: {sorted(modelos_validos)}"
        )

    X_train = _as_numpy(X_train)
    X_val = _as_numpy(X_val)

    y_train = np.asarray(y_train).ravel()
    y_val = np.asarray(y_val).ravel()

    if nombre_modelo in {"autoencoder", "vae"}:
        X_fit = X_train[y_train == 0]
    else:
        X_fit = X_train

    def objective(trial):
        try:
            modelo = _crear_modelo_optuna(
                trial=trial,
                nombre_modelo=nombre_modelo,
                random_state=random_state,
            )

            modelo.fit(X_fit)

            y_pred_val = modelo.predict(X_val)

            return f1_score(
                y_val,
                y_pred_val,
                zero_division=0,
            )

        except RuntimeWarning as error:
            print(
                f"Trial {trial.number} podado por "
                f"inestabilidad numérica de MCD."
            )

            trial.set_user_attr(
                "error",
                str(error),
            )

            raise optuna.TrialPruned()

        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as error:
            print(
                f"Trial {trial.number} podado: {error}"
            )

            trial.set_user_attr(
                "error",
                str(error),
            )

            raise optuna.TrialPruned()
        
    study = crear_estudio(
        dataset=dataset_name,
        nombre_modelo=nombre_modelo,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[
            crear_callback_guardado(
                dataset=dataset_name,
                nombre_modelo=nombre_modelo,
            )
        ],
        gc_after_trial=True,
    )

    print(
        f"\nMejor F1 de validación para {nombre_modelo}: "
        f"{study.best_value:.6f}"
    )

    print("Mejores hiperparámetros:")

    for parametro, valor in study.best_params.items():
        print(f"  {parametro}: {valor}")

    return study
