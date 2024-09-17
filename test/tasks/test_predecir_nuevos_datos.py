import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from config.constants import MODELS_DIR, DATA_DIR, NUEVOS_CLIENTES_FILENAME, NUEVOS_CLIENTES_PREDICCION_FILENAME, \
    BEST_MODEL_FILENAME, IMPUTER_FILENAME, SCALER_FILENAME
from tasks.predecir_nuevos_datos import predecir_nuevos_datos


class TestPredecirNuevosDatos(unittest.TestCase):

    @patch('tasks.predecir_nuevos_datos.joblib.load')
    @patch('tasks.predecir_nuevos_datos.pd.read_csv')
    @patch('tasks.predecir_nuevos_datos.pd.DataFrame.to_csv')
    def test_predecir_nuevos_datos(self, mock_to_csv, mock_read_csv, mock_joblib_load):
        mock_best_model = MagicMock()
        mock_best_model.predict.return_value = [0, 1, 0]

        mock_imputer = MagicMock()
        mock_imputer.transform.return_value = np.array([[1, 2], [3, 4], [5, 6]])

        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        mock_joblib_load.side_effect = [mock_best_model, mock_imputer, mock_scaler]

        mock_read_csv.return_value = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'categoria': ['A', 'B', 'A']
        })

        predecir_nuevos_datos()

        mock_joblib_load.assert_any_call(os.path.join(MODELS_DIR, BEST_MODEL_FILENAME))
        mock_joblib_load.assert_any_call(os.path.join(MODELS_DIR, IMPUTER_FILENAME))
        mock_joblib_load.assert_any_call(os.path.join(MODELS_DIR, SCALER_FILENAME))

        mock_read_csv.assert_called_once_with(os.path.join(DATA_DIR, NUEVOS_CLIENTES_FILENAME))

        mock_to_csv.assert_called_once_with(os.path.join(DATA_DIR, NUEVOS_CLIENTES_PREDICCION_FILENAME), index=False)
