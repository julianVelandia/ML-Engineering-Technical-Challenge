import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

from config.constants import DATA_DIR, MODELS_DIR, CLEAN_FEATURES_FILENAME, CLEAN_TARGETS_FILENAME, \
    BEST_METRICS_FILENAME, BEST_MODEL_FILENAME


def ajustar_hiperparametros():
    """
    Ajusta los hiperparámetros de un modelo RandomForestClassifier mediante GridSearchCV,
    entrena el mejor modelo encontrado y guarda tanto las métricas del modelo como el modelo entrenado.
    """

    X = pd.read_csv(os.path.join(DATA_DIR, CLEAN_FEATURES_FILENAME))
    y = pd.read_csv(os.path.join(DATA_DIR, CLEAN_TARGETS_FILENAME))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='f1_weighted',
        verbose=2
    )

    grid_search.fit(X_train, y_train.values.ravel())

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    best_metrics = {
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall']
    }

    with open(os.path.join(MODELS_DIR, BEST_METRICS_FILENAME), 'w') as f:
        json.dump(best_metrics, f)

    joblib.dump(best_model, os.path.join(MODELS_DIR, BEST_MODEL_FILENAME))
