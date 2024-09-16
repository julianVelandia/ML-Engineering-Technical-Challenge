import os

from config.constants import DATA_DIR, FEATURES_FILENAME, TARGETS_FILENAME
from ucimlrepo import fetch_ucirepo


def cargar_datos_ucirepo(dataset_id: int):
    """
        Descarga los datos de un dataset específico desde UC Repository, los guarda en archivos CSV, y
        los almacena en el directorio de datos definido en las constantes.

        Args:
            dataset_id (int): El ID del dataset que se desea descargar desde UC Irvine.

        Proceso:
            - Crea el directorio de datos si no existe.
            - Descarga los datos de características y objetivos.
            - Guarda los datos en archivos CSV en el directorio de datos.
    """
    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features
    y = dataset.data.targets

    os.makedirs(DATA_DIR, exist_ok=True)

    X.to_csv(os.path.join(DATA_DIR, FEATURES_FILENAME), index=False)
    y.to_csv(os.path.join(DATA_DIR, TARGETS_FILENAME), index=False)
