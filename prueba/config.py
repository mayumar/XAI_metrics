# src/config.py

import os
from datetime import timedelta
import pandas as pd

# ======================
# RUTAS GENERALES
# ======================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
INPUT_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
SHAP_DIR = os.path.join(OUTPUT_DIR, "SHAP_plots")
LIME_DIR = os.path.join(OUTPUT_DIR, "LIME_plots")
PDP_DIR = os.path.join(OUTPUT_DIR, "PDP_plots")

RANDOM_STATE = 42
TEST_SIZE = 0.4   # para datasets sin división temporal
CONTAMINATION = None  # si queremos forzar una fracción fija de anomalías
NR_SAMPLES = 20

# ======================
# PARÁMETROS ESPECÍFICOS POR DATASET
# ======================
DATASETS = {
    "metro": {
        "ruta_origen": os.path.join(INPUT_DIR, "MetroPT3(AirCompressor).csv"),
        # Columnas a eliminar despues de correlacion
        "variables_eliminar": ['DV_eletric', 'TP3', 'H1', 'COMP', 'MPG'],
        # Lista de intervalos: cada tupla es (inicio, fin)
        "intervalos": [
            (pd.Timestamp("2020-04-12 11:50:00"), pd.Timestamp("2020-04-12 23:30:00")),# Train
            (pd.Timestamp("2020-04-18 00:00:00"), pd.Timestamp("2020-04-19 01:30:00")),#
            (pd.Timestamp("2020-04-29 03:20:00"), pd.Timestamp("2020-04-29 04:00:00")),
            (pd.Timestamp("2020-04-29 22:00:00"), pd.Timestamp("2020-04-29 22:20:00")),
            (pd.Timestamp("2020-05-13 14:00:00"), pd.Timestamp("2020-05-13 23:59:00")),
            (pd.Timestamp("2020-05-18 05:00:00"), pd.Timestamp("2020-05-18 05:30:00")),
            (pd.Timestamp("2020-05-19 10:10:00"), pd.Timestamp("2020-05-19 11:00:00")),
            (pd.Timestamp("2020-05-19 22:10:00"), pd.Timestamp("2020-05-20 20:00:00")),
            (pd.Timestamp("2020-05-23 09:50:00"), pd.Timestamp("2020-05-23 10:10:00")),
            (pd.Timestamp("2020-05-29 23:30:00"), pd.Timestamp("2020-05-30 06:00:00")),#
            (pd.Timestamp("2020-06-01 15:00:00"), pd.Timestamp("2020-06-01 15:40:00")),
            (pd.Timestamp("2020-06-03 10:00:00"), pd.Timestamp("2020-06-03 11:00:00")),
            (pd.Timestamp("2020-06-05 10:00:00"), pd.Timestamp("2020-06-07 14:30:00")),#
            (pd.Timestamp("2020-07-08 17:30:00"), pd.Timestamp("2020-07-08 19:00:00")),# <------ (1h30) Test empieza aquí
            (pd.Timestamp("2020-07-15 14:30:00"), pd.Timestamp("2020-07-15 19:00:00")),# <------ (4h30)
            (pd.Timestamp("2020-07-17 04:30:00"), pd.Timestamp("2020-07-17 05:30:00"))
        ],
        "observations": [
            pd.Timestamp("2020-04-18 00:00:00"), # --> Fallan todos
            pd.Timestamp("2020-04-18 02:30:00"), # --> AutoEncoder
            pd.Timestamp("2020-04-18 06:00:00"), # --> IForest + HBOS
            
            pd.Timestamp("2020-05-18 05:00:00"), # --> Aciertan casi todos
            pd.Timestamp("2020-05-18 05:30:00"), # --> No acierta ni dios
            
            # pd.Timestamp("2020-05-20 02:00:00"),
            # pd.Timestamp("2020-05-20 02:30:00"),
            # pd.Timestamp("2020-05-20 09:30:00"), # --> No acierta AutoEncoder ni MCD

            pd.Timestamp("2020-05-13 20:00:00"),
            pd.Timestamp("2020-05-13 20:30:00"),
            pd.Timestamp("2020-05-13 21:00:00"),

            # pd.Timestamp("2020-06-01 15:00:00"),
            # pd.Timestamp("2020-06-01 15:30:00")
        ],
        # Frecuencia de resampleo
        "resample_freq": "30min",
        # Metodo de rellenado
        "fill_method": "ffill",
    },
    "hydraulic": {
        "ruta_origen": "/home/mayumar/Escritorio/Pruebas/datasets/condition+monitoring+of+hydraulic+systems",
        "sensores": ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE'],
        "error_types": ['cooler','valve','leakage','accumulator','stable'],
        "target": {"cooler": 3},
        # Columnas a eliminar despues de correlacion
        "variables_eliminar": ['PS2', 'FS1', 'SE', 'PS6', 'CE', 'CP', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4'], 
        "observations": [
            9,
            5,
            249,
            252,
            255,
            685,
            403,
            420
        ]
    },
}
