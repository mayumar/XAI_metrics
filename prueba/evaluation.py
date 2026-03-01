# src/evaluation.py

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from config import DATASETS
import matplotlib.pyplot as plt

def evaluar_modelo(metricas, clf_name, y_test, prediction, normalizado, contaminacion, duracion, semilla):
    # Calculamos las metricas para test
    cm = confusion_matrix(y_test, prediction)
    print('Matriz de confusión: \n', cm)

    # Métricas personalizadas desde matriz de confusión
    sensibilidad = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    especificidad = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0

    # Métricas generales
    roc_auc = roc_auc_score(y_test, prediction)
    acc = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    # Añadir resultados al DataFrame
    nueva_fila = {
        'Modelo': clf_name,
        'Semilla': semilla,
        'Normalizado': normalizado,
        'Contaminacion': contaminacion,
        'TN' : cm[0,0],
        'FP' : cm[0,1],
        'FN' : cm[1,0],
        'TP' : cm[1,1],
        'Accuracy': acc,
        'F1-score': f1,
        'Sensibilidad': sensibilidad,
        'Especificidad': especificidad,
        'Precisión': precision,
        'ROC-AUC': roc_auc,
        'Tiempo (s)': duracion
    }
    nueva_df = pd.DataFrame([nueva_fila])

    # Si metricas está vacío, lo reemplazamos; si no, concatenamos
    if metricas.empty:
        metricas = nueva_df
    else:
        metricas = pd.concat([metricas, nueva_df], ignore_index=True)
        
    return metricas

def representar_fallos(clf, clf_name, dataset_name, X_test, y_test):
    prediction = clf.predict(X_test)

    # Dataset basado en series temporales
    if dataset_name == "metro":
        print('\n*** Eventos de anomalía y sus FN ***')
        for inicio, fin in DATASETS[dataset_name]["intervalos"]:
            # Crear el DataFrame combinando los valores reales y predichos
            df = pd.DataFrame({
                'Real': y_test,
                'Predicho': prediction
            }, index=y_test.index)

            # Filtrar errores
            df_errores_inicial = df[df['Real'] != df['Predicho']]

            # Filtrar por rango de fechas
            df_errores = df_errores_inicial.loc[inicio:fin]

            if not df_errores.empty:
                print(f"Evento: {inicio} / {fin}")

                print(df_errores.to_string())

                # Graficar los errores
                plt.figure(figsize=(12, 5))
                plt.scatter(df_errores.index, df_errores['Real'], color='blue', alpha=0.6, label='Real', marker='o')
                plt.scatter(df_errores.index, df_errores['Predicho'], color='red', alpha=0.4, label='Predicho', marker='x')

                plt.title(f'Errores de Clasificación - {clf_name}')
                plt.xlabel('Fecha')
                plt.ylabel('Clase')
                plt.yticks([0, 1])
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.xticks(rotation=45)

                plt.xlim(pd.to_datetime(inicio), pd.to_datetime(fin))
                
                plt.show()
    else:
        print('\n*** Anomalías y sus FN ***')

        # Crear el DataFrame combinando los valores reales y predichos
        df = pd.DataFrame({
            'Real': y_test,
            'Predicho': prediction
        }, index=y_test.index)

        # Filtrar errores
        df_errores = df[(df['Real'] == 1) & (df['Predicho'] == 0)].sort_index()

        if not df_errores.empty:
            print(df_errores.to_string())

            # Graficar los errores
            plt.figure(figsize=(12, 5))
            plt.scatter(df_errores.index, df_errores['Real'], color='blue', alpha=0.6, label='Real', marker='o')
            plt.scatter(df_errores.index, df_errores['Predicho'], color='red', alpha=0.4, label='Predicho', marker='x')

            plt.title(f'Errores de Clasificación - {clf_name}')
            plt.xlabel('Indice')
            plt.ylabel('Clase')
            plt.yticks([0, 1])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.xticks(rotation=45)
            
            plt.show()