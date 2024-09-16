import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from config.constants import CLEAN_FEATURES_FILENAME, CLEAN_TARGETS_FILENAME, MODELS_DIR, INITIAL_METRICS_FILENAME, \
    MODEL_FILENAME, DATA_DIR


def entrenar_modelo():
    """
        Función que entrena un modelo de clasificación Random Forest utilizando los datos preprocesados.

        Carga los datos limpios de características y etiquetas, realiza la división de los datos en conjuntos de
        entrenamiento y prueba, entrena el modelo de Random Forest y luego guarda tanto las métricas iniciales del
        modelo como el modelo entrenado en archivos JSON y joblib, respectivamente.

        Archivos generados:
        - initial_metrics.json: Un archivo JSON que contiene las métricas F1, precisión y recall del modelo entrenado.
        - random_forest_model.joblib: Archivo que contiene el modelo entrenado de Random Forest.
    """
    X = pd.read_csv(os.path.join(DATA_DIR, CLEAN_FEATURES_FILENAME))
    y = pd.read_csv(os.path.join(DATA_DIR, CLEAN_TARGETS_FILENAME))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))

    initial_metrics = {
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall']
    }
    os.makedirs(MODELS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, INITIAL_METRICS_FILENAME), 'w') as f:
        json.dump(initial_metrics, f)

    joblib.dump(model, os.path.join(MODELS_DIR, MODEL_FILENAME))
    print("Modelo entrenado y guardado exitosamente.")
