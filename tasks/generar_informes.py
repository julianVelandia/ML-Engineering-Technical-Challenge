import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

from config.constants import MODELS_DIR, DATA_DIR, REPORTS_DIR, CLEAN_FEATURES_FILENAME, CLEAN_TARGETS_FILENAME, \
    BEST_MODEL_FILENAME


def generar_informes():
    """
    Genera informes de desempeño del modelo entrenado, incluyendo una matriz de confusión y la curva ROC,
    y los guarda como imágenes en un directorio específico de reportes.

    Proceso:
        - Carga el modelo entrenado desde el archivo correspondiente.
        - Carga los datos de características y objetivos.
        - Realiza predicciones y calcula probabilidades.
        - Genera una matriz de confusión y una curva ROC.
        - Guarda las imágenes generadas en la carpeta de reportes.
    """
    best_model = joblib.load(os.path.join(MODELS_DIR, BEST_MODEL_FILENAME))

    X = pd.read_csv(os.path.join(DATA_DIR, CLEAN_FEATURES_FILENAME))
    y = pd.read_csv(os.path.join(DATA_DIR, CLEAN_TARGETS_FILENAME))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = best_model.predict(X_test)
    y_prob = np.array(best_model.predict_proba(X_test))[:, 1]

    os.makedirs(REPORTS_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
    plt.close()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'roc_curve.png'))
    plt.close()
