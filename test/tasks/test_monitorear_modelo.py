import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from config.constants import DATA_DIR, NUEVOS_DATOS_FILENAME, DATOS_ANTERIORES_FILENAME
from tasks.monitorear_modelo import monitorear_modelo


class TestMonitorearModelo(unittest.TestCase):

    @patch('tasks.monitorear_modelo.joblib.load')
    @patch('tasks.monitorear_modelo.pd.read_csv')
    @patch('tasks.monitorear_modelo.json.dump')
    @patch('tasks.monitorear_modelo.json.load')
    @patch('tasks.monitorear_modelo.os.path.exists')
    @patch('tasks.monitorear_modelo.open', create=True)
    def test_monitorear_modelo(self, mock_open, mock_exists, mock_load_json, mock_dump_json, mock_read_csv,
                               mock_joblib_load):
        mock_best_model = MagicMock()
        mock_imputer = MagicMock()
        mock_scaler = MagicMock()

        mock_imputer.transform.return_value = np.array([[1, 2], [3, 4], [5, 6]])
        mock_scaler.transform.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        mock_joblib_load.side_effect = [mock_best_model, mock_imputer, mock_scaler]

        mock_exists.side_effect = [True, True]

        mock_read_csv.return_value = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'target': [0, 1, 0]
        })

        mock_best_model.predict.return_value = np.array([0, 1, 0])
        mock_best_model.predict_proba.return_value = np.array([[0.6, 0.4], [0.2, 0.8], [0.7, 0.3]])

        mock_load_json.return_value = {'f1_score': 0.95}

        monitorear_modelo()

        mock_exists.assert_any_call(os.path.join(DATA_DIR, NUEVOS_DATOS_FILENAME))
        mock_read_csv.assert_called_with(os.path.join(DATA_DIR, NUEVOS_DATOS_FILENAME))

        mock_dump_json.assert_called()

    @patch('tasks.monitorear_modelo.joblib.load')
    @patch('tasks.monitorear_modelo.pd.read_csv')
    @patch('tasks.monitorear_modelo.json.dump')
    @patch('tasks.monitorear_modelo.json.load')
    @patch('tasks.monitorear_modelo.os.path.exists')
    @patch('tasks.monitorear_modelo.open', create=True)
    def test_monitorear_modelo_sin_nuevos_datos(self, mock_open, mock_exists, mock_load_json, mock_dump_json,
                                                mock_read_csv, mock_joblib_load):
        mock_best_model = MagicMock()
        mock_imputer = MagicMock()
        mock_scaler = MagicMock()
        mock_imputer.transform.return_value = np.array([[1, 2], [3, 4], [5, 6]])
        mock_scaler.transform.return_value = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Simular salida correcta

        mock_joblib_load.side_effect = [mock_best_model, mock_imputer, mock_scaler]
        mock_exists.side_effect = [False, True]

        mock_read_csv.return_value = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'target': [0, 1, 0]
        })

        mock_load_json.return_value = {'f1_score': 0.95}

        monitorear_modelo()

        mock_exists.assert_any_call(os.path.join(DATA_DIR, NUEVOS_DATOS_FILENAME))
        mock_read_csv.assert_called_with(os.path.join(DATA_DIR, DATOS_ANTERIORES_FILENAME))

        mock_dump_json.assert_called()
