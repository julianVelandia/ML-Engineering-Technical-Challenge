import os

from config.constants import MODELS_DIR, RETRAIN_FLAG_FILENAME
from tasks.ajustar_hiperparametros import ajustar_hiperparametros


def reentrenar_modelo():
    """
        Reentrena el modelo si se encuentra el archivo de marca para reentrenamiento.

        Proceso:
            - Verifica si existe el archivo de reentrenamiento en la ruta definida.
            - Si el archivo existe, lo elimina y llama a la función de ajuste de hiperparámetros.
            - Si no existe, imprime un mensaje indicando que no se requiere reentrenamiento.

        Returns:
            None
    """
    retrain_flag_path = os.path.join(MODELS_DIR, RETRAIN_FLAG_FILENAME)

    if os.path.exists(retrain_flag_path):
        os.remove(retrain_flag_path)
        ajustar_hiperparametros()
