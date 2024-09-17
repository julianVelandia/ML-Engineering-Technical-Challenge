import os

import joblib
import numpy as np
import pandas as pd

from config.constants import MODELS_DIR, DATA_DIR, NUEVOS_CLIENTES_FILENAME, NUEVOS_CLIENTES_PREDICCION_FILENAME, \
    BEST_MODEL_FILENAME, IMPUTER_FILENAME, SCALER_FILENAME


def predecir_nuevos_datos():
    """
    Esta función carga el mejor modelo entrenado, junto con un imputador y escalador para predecir
    el comportamiento de nuevos clientes. La predicción se guarda en un archivo CSV.

    El proceso incluye:
    - Cargar el modelo, el imputador y el escalador.
    - Cargar los datos de nuevos clientes desde un archivo CSV.
    - Imputar los valores faltantes y escalar las características numéricas.
    - Realizar predicciones usando el modelo cargado.
    - Guardar los resultados en un nuevo archivo CSV.
    """
    best_model = joblib.load(os.path.join(MODELS_DIR, BEST_MODEL_FILENAME))
    imputer = joblib.load(os.path.join(MODELS_DIR, IMPUTER_FILENAME))
    scaler = joblib.load(os.path.join(MODELS_DIR, SCALER_FILENAME))

    nuevos_datos = pd.read_csv(os.path.join(DATA_DIR, NUEVOS_CLIENTES_FILENAME))

    numeric_columns = nuevos_datos.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = nuevos_datos.select_dtypes(exclude=np.number).columns.tolist()

    nuevos_datos_numeric = pd.DataFrame(imputer.transform(nuevos_datos[numeric_columns]), columns=numeric_columns)

    nuevos_datos_combined = pd.concat([nuevos_datos_numeric, nuevos_datos[categorical_columns]], axis=1)

    nuevos_datos_combined[numeric_columns] = scaler.transform(nuevos_datos_combined[numeric_columns])

    predicciones = best_model.predict(nuevos_datos_combined)

    nuevos_datos['prediccion_default'] = predicciones

    nuevos_datos.to_csv(os.path.join(DATA_DIR, NUEVOS_CLIENTES_PREDICCION_FILENAME), index=False)
