# src/data_processing.py

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from config import DATASETS, TEST_SIZE

def load_data_metro(show_plot):
    print("--- Procesando dataset MetroPT-3 ---")
    df = pd.read_csv(DATASETS["metro"]["ruta_origen"])
    df.drop(df.columns[0], axis=1, inplace=True)

    if show_plot:
        show_correlation_matrix(df.drop('timestamp', axis=1))

    df = df.drop(DATASETS["metro"]["variables_eliminar"], axis=1, inplace=False) # Aqui se han quedado Reservoirs y TP2

    # Nos aseguramos de que 'timestamp' es de tipo datetime y el índice
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Remuestrea a intervalos de 1 hora, aplicando una función agregada (por ejemplo, la media)
    df = df.asfreq(DATASETS["metro"]["resample_freq"], method=DATASETS["metro"]["fill_method"])

    # Se ordena el dataset cronologicamente
    df = df.sort_index()

    # Inicializamos la columna binaria en 0
    df['failure'] = 0

    # Iteramos sobre cada intervalo y asignamos 1 a los rangos correspondientes
    for inicio, fin in DATASETS["metro"]["intervalos"]:
        df.loc[(df.index >= inicio) & (df.index <= fin), 'failure'] = 1


    # Calcular el índice de división para obtener el 60% de los datos
    split_index = int((1 - TEST_SIZE) * len(df))

    # Dividir en conjuntos de entrenamiento y prueba
    train_dataset = df.iloc[:split_index]
    test_dataset = df.iloc[split_index:]

    X_train = train_dataset.drop('failure', axis=1, inplace=False)
    y_train = train_dataset['failure']

    X_test = test_dataset.drop('failure', axis=1, inplace=False)
    y_test = test_dataset['failure']

    return X_train, X_test, y_train, y_test

def load_data_hydraulic_systems(show_plot):
    print("--- Procesando dataset Condition monitoring of hydraulic systems ---")
    medias = {}

    for sensor in DATASETS["hydraulic"]["sensores"]:
        file_path = os.path.join(DATASETS["hydraulic"]["ruta_origen"], f"{sensor}.txt")
        df_sensor = pd.read_csv(file_path, sep="\t", header=None)
        # (axis=1 → media horizontal en cada muestra)
        medias[sensor] = df_sensor.mean(axis=1)

    df = pd.DataFrame(medias)

    profile_path = os.path.join(DATASETS["hydraulic"]["ruta_origen"], "profile.txt")
    df_profile = pd.read_csv(profile_path, sep="\t", header=None, names=DATASETS["hydraulic"]["error_types"])

    # Definimos target binario
    optimal = (
        (df_profile['cooler'] == DATASETS["hydraulic"]["target"]["cooler"])
    )

    # Si alguna no óptima → 0, si no → 1
    df['Target'] = optimal.astype(int)

    if show_plot:
        show_correlation_matrix(df.drop('Target', axis=1))

    df = df.drop(DATASETS["hydraulic"]["variables_eliminar"], axis=1, inplace=False) # Aqui se han quedado PS1 y PS5

    X = df.drop(['Target'], axis=1)
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=y
        )

    return X_train, X_test, y_train, y_test

def preprocess_dataset(dataset_name, show_plot):
    # 1) Carga específica según dataset_name
    if dataset_name == "metro":
        X_train, X_test, y_train, y_test = load_data_metro(show_plot)
    elif dataset_name == "hydraulic":
        X_train, X_test, y_train, y_test = load_data_hydraulic_systems(show_plot)
    else:
        raise ValueError(f"Dataset desconocido: {dataset_name}")
    
    # Fracción de anomalías que tiene el conjunto de datos
    y = pd.concat([y_train, y_test], ignore_index=True)

    anomalias_fraccion = np.count_nonzero(y.values) / len(y.values)
    print("Fracción de anomalías:", anomalias_fraccion)

    anomalias_porcentaje = round(anomalias_fraccion * 100, ndigits=4)
    print("Porcentaje de anomalías:", anomalias_porcentaje)

    # Normalizamos los datos en el itervalo [0, 1]
    scaler = MinMaxScaler()

    X_train_norm = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test),   # <-- transform, no fit_transform
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train, y_train, X_test, y_test, X_train_norm, X_test_norm, anomalias_fraccion

def show_correlation_matrix(X):
    correlation_matrix = X.corr()
    # print(correlation_matrix)
    # plt.figure(figsize=(20,17))
    sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True)
    plt.title('Correlation Matrix')
    plt.show()
