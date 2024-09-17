import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from config.constants import MODELS_DIR, DATA_DIR, CURRENT_METRICS_FILENAME, BEST_METRICS_FILENAME, \
    RETRAIN_FLAG_FILENAME, \
    RANDOM_FOREST_MODEL_FILENAME, IMPUTER_FILENAME, SCALER_FILENAME, NUEVOS_DATOS_FILENAME, CLEAN_FEATURES_FILENAME, \
    CLEAN_TARGETS_FILENAME


def monitorear_modelo():
    """
    Monitorea el desempeño del modelo en nuevos datos etiquetados y compara las métricas actuales con las mejores
    métricas previamente almacenadas. Si no se encuentran nuevos datos, utiliza los datos anteriores. Si el rendimiento
    del modelo ha disminuido significativamente, se marca para reentrenamiento.

    Proceso:
        - Carga el modelo entrenado, el imputador y el escalador.
        - Carga nuevos datos etiquetados o, si no existen, los datos anteriores.
        - Preprocesa los datos y genera predicciones.
        - Calcula las métricas y las compara con las mejores métricas almacenadas.
        - Si el f1_score actual es menor al 95% del mejor f1_score, marca el modelo para reentrenamiento.
    """
    best_model = joblib.load(os.path.join(MODELS_DIR, RANDOM_FOREST_MODEL_FILENAME))
    imputer = joblib.load(os.path.join(MODELS_DIR, IMPUTER_FILENAME))
    scaler = joblib.load(os.path.join(MODELS_DIR, SCALER_FILENAME))

    if os.path.exists(os.path.join(DATA_DIR, NUEVOS_DATOS_FILENAME)):
        datos = pd.read_csv(os.path.join(DATA_DIR, NUEVOS_DATOS_FILENAME))
    else:
        datos = pd.read_csv(os.path.join(DATA_DIR, CLEAN_FEATURES_FILENAME))
        y_true = pd.read_csv(os.path.join(DATA_DIR, CLEAN_TARGETS_FILENAME))

    y_true = datos['target'] if 'target' in datos.columns else y_true
    X_new = datos.drop('target', axis=1)

    numeric_columns = X_new.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X_new.select_dtypes(exclude=np.number).columns.tolist()

    X_numeric = pd.DataFrame(imputer.transform(X_new[numeric_columns]), columns=numeric_columns)
    X_combined = pd.concat([X_numeric, X_new[categorical_columns]], axis=1)
    X_combined[numeric_columns] = scaler.transform(X_combined[numeric_columns])

    y_pred = best_model.predict(X_combined)

    f1 = f1_score(y_true, y_pred, average='weighted')

    current_metrics = {
        'f1_score': f1,
        'precision': f1_score(y_true, y_pred, average='weighted'),
        'recall': f1_score(y_true, y_pred, average='weighted')
    }

    with open(os.path.join(MODELS_DIR, CURRENT_METRICS_FILENAME), 'w') as f:
        json.dump(current_metrics, f)

    with open(os.path.join(MODELS_DIR, BEST_METRICS_FILENAME), 'r') as f:
        best_metrics = json.load(f)

    if f1 < best_metrics['f1_score'] * 0.95:
        with open(os.path.join(MODELS_DIR, RETRAIN_FLAG_FILENAME), 'w') as f:
            f.write('retrain')
