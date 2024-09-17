import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from config.constants import DATA_DIR, MODELS_DIR, FEATURES_FILENAME, TARGETS_FILENAME, CLEAN_FEATURES_FILENAME, \
    CLEAN_TARGETS_FILENAME, IMPUTER_FILENAME, SCALER_FILENAME


def limpiar_datos():
    """
    Función para limpiar los datos del dataset .

    Pasos:
    1. Carga los datos desde archivos CSV.
    2. Imputa valores faltantes en las columnas numéricas usando KNNImputer.
    3. Elimina outliers usando el método del rango intercuartil (IQR).
    4. Escala las columnas numéricas utilizando StandardScaler.
    5. Guarda los datos limpiados y los modelos de imputación y escalado.
    """
    X = pd.read_csv(os.path.join(DATA_DIR, FEATURES_FILENAME))
    y = pd.read_csv(os.path.join(DATA_DIR, TARGETS_FILENAME))

    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=np.number).columns.tolist()

    imputer = KNNImputer(n_neighbors=5)
    X_numeric = pd.DataFrame(imputer.fit_transform(X[numeric_columns]), columns=numeric_columns)

    X_combined = pd.concat([X_numeric, X[categorical_columns]], axis=1)

    Q1 = X_numeric.quantile(0.25)
    Q3 = X_numeric.quantile(0.75)
    IQR = Q3 - Q1
    umbral = 1.5
    outlier_condition = ~((X_numeric < (Q1 - umbral * IQR)) | (X_numeric > (Q3 + umbral * IQR))).any(axis=1)
    X_clean = X_combined.loc[outlier_condition]
    y_clean = y.loc[outlier_condition]

    scaler = StandardScaler()
    X_clean[numeric_columns] = scaler.fit_transform(X_clean[numeric_columns])

    X_clean.to_csv(os.path.join(DATA_DIR, CLEAN_FEATURES_FILENAME), index=False)
    y_clean.to_csv(os.path.join(DATA_DIR, CLEAN_TARGETS_FILENAME), index=False)
    joblib.dump(imputer, os.path.join(MODELS_DIR, IMPUTER_FILENAME))
    joblib.dump(scaler, os.path.join(MODELS_DIR, SCALER_FILENAME))
