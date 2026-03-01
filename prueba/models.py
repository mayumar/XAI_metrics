# src/models.py

from time import time
from pyod.models.cblof import CBLOF
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.hbos import HBOS
from pyod.models.mcd import MCD
from pyod.models.vae import VAE

from evaluation import evaluar_modelo  # Importar la función de métricas

def usar_cblof(X_train, y_train, X_test, y_test, metricas, normalizado, contaminacion, random_state):
    # Inicializamos el tiempo
    t0 = time()

    # Entrenamos CBLOF
    if contaminacion == None:
        clf_CBLOF = CBLOF(random_state=random_state)
    else:
        clf_CBLOF = CBLOF(contamination=contaminacion,random_state=random_state)
    clf_CBLOF.fit(X_train)

    # Obtenemos la prediccion
    y_pred = clf_CBLOF.predict(X_test)

    # Una vez tenemos los resultados, finalizamos el tiempo
    t1 = time()
    duracion = round(t1 - t0, ndigits=4)

    metricas_result = evaluar_modelo(metricas, 'CBLOF', y_test, y_pred, normalizado, contaminacion, duracion, random_state)
    return metricas_result, clf_CBLOF

def usar_iforest(X_train, y_train, X_test, y_test, metricas, normalizado, contaminacion, random_state):
    # Inicializamos el tiempo
    t0 = time()

    # Entrenamos Iforest
    if contaminacion == None:
        clf_IForest = IForest(random_state=random_state)
    else:
        clf_IForest = IForest(contamination=contaminacion,random_state=random_state)
    clf_IForest.fit(X_train)

    # Obtenemos la prediccion
    y_pred = clf_IForest.predict(X_test)

    # Una vez tenemos los resultados, finalizamos el tiempo
    t1 = time()
    duracion = round(t1 - t0, ndigits=4)

    metricas_result = evaluar_modelo(metricas, 'IForest', y_test, y_pred, normalizado, contaminacion, duracion, random_state)
    return metricas_result, clf_IForest

def usar_ecod(X_train, y_train, X_test, y_test, metricas, normalizado, contaminacion, random_state):
    # Inicializamos el tiempo
    t0 = time()

    # Entrenamos ECOD
    if contaminacion == None:
        clf_ECOD = ECOD()
    else:
        clf_ECOD = ECOD(contamination=contaminacion)
    clf_ECOD.fit(X_train)

    # Obtenemos la prediccion
    y_pred = clf_ECOD.predict(X_test)

    # Una vez tenemos los resultados, finalizamos el tiempo
    t1 = time()
    duracion = round(t1 - t0, ndigits=4)

    metricas_result = evaluar_modelo(metricas, 'ECOD', y_test, y_pred, normalizado, contaminacion, duracion, random_state)
    return metricas_result, clf_ECOD

def usar_autoencoder(X_train, y_train, X_test, y_test, metricas, normalizado, contaminacion, random_state):
    # Inicializamos el tiempo
    t0 = time()

    # Entrenamos AutoEncoder
    if contaminacion == None:
        clf_AutoEncoder = AutoEncoder(random_state=random_state)
    else:
        clf_AutoEncoder = AutoEncoder(contamination=contaminacion,random_state=random_state)
    X_train_no_failures = X_train[y_train == 0]
    clf_AutoEncoder.fit(X_train_no_failures)

    # Obtenemos la prediccion
    y_pred = clf_AutoEncoder.predict(X_test)

    # Una vez tenemos los resultados, finalizamos el tiempo
    t1 = time()
    duracion = round(t1 - t0, ndigits=4)

    metricas_result = evaluar_modelo(metricas, 'AutoEncoder', y_test, y_pred, normalizado, contaminacion, duracion, random_state)
    return metricas_result, clf_AutoEncoder

def usar_hbos(X_train, y_train, X_test, y_test, metricas, normalizado, contaminacion, random_state):
    # Inicializamos el tiempo
    t0 = time()

    # Entrenamos HBOS
    if contaminacion == None:
        clf_HBOS = HBOS()
    else:
        clf_HBOS = HBOS(contamination=contaminacion)
    clf_HBOS.fit(X_train)

    # Obtenemos la prediccion
    y_pred = clf_HBOS.predict(X_test)

    # Una vez tenemos los resultados, finalizamos el tiempo
    t1 = time()
    duracion = round(t1 - t0, ndigits=4)

    metricas_result = evaluar_modelo(metricas, 'HBOS', y_test, y_pred, normalizado, contaminacion, duracion, random_state)
    return metricas_result, clf_HBOS

def usar_mcd(X_train, y_train, X_test, y_test, metricas, normalizado, contaminacion, random_state):
    # Inicializamos el tiempo
    t0 = time()

    # Entrenamos MCD
    if contaminacion == None:
        clf_MCD = MCD(random_state=random_state)
    else:
        clf_MCD = MCD(contamination=contaminacion,random_state=random_state)
    clf_MCD.fit(X_train)

    # Obtenemos la prediccion
    y_pred = clf_MCD.predict(X_test)

    # Una vez tenemos los resultados, finalizamos el tiempo
    t1 = time()
    duracion = round(t1 - t0, ndigits=4)

    metricas_result = evaluar_modelo(metricas, 'MCD', y_test, y_pred, normalizado, contaminacion, duracion, random_state)
    return metricas_result, clf_MCD

def usar_vae(X_train, y_train, X_test, y_test, metricas, normalizado, contaminacion, random_state):
    # Inicializamos el tiempo
    t0 = time()

    # Entrenamos VAE
    if contaminacion == None:
        clf_VAE = VAE(random_state=random_state)
    else:
        clf_VAE = VAE(contamination=contaminacion,random_state=random_state)
    X_train_no_failures = X_train[y_train == 0]
    clf_VAE.fit(X_train_no_failures)

    # Obtenemos la prediccion
    y_pred = clf_VAE.predict(X_test)

    # Una vez tenemos los resultados, finalizamos el tiempo
    t1 = time()
    duracion = round(t1 - t0, ndigits=4)

    metricas_result = evaluar_modelo(metricas, 'VAE', y_test, y_pred, normalizado, contaminacion, duracion, random_state)
    return metricas_result, clf_VAE