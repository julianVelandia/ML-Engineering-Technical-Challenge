import os
import unittest
from unittest.mock import patch

import pandas as pd

from config.constants import DATA_DIR, MODELS_DIR, FEATURES_FILENAME, TARGETS_FILENAME, CLEAN_FEATURES_FILENAME, \
    CLEAN_TARGETS_FILENAME, IMPUTER_FILENAME, SCALER_FILENAME
from tasks.limpiar_datos import limpiar_datos


class TestLimpiarDatos(unittest.TestCase):

    @patch('tasks.limpiar_datos.joblib.dump')
    @patch('tasks.limpiar_datos.pd.DataFrame.to_csv')
    @patch('tasks.limpiar_datos.StandardScaler.fit_transform')
    @patch('tasks.limpiar_datos.KNNImputer.fit_transform')
    @patch('tasks.limpiar_datos.pd.read_csv')
    def test_limpiar_datos(self, mock_read_csv, mock_knn_imputer, mock_scaler, mock_to_csv, mock_joblib_dump):
        mock_read_csv.side_effect = [
            pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}),
            pd.DataFrame({'target': [0, 1]})
        ]

        mock_knn_imputer.return_value = [[1.0, 2.0], [3.0, 4.0]]
        mock_scaler.return_value = [[0.0, 1.0], [1.5, 2.5]]

        limpiar_datos()

        mock_read_csv.assert_any_call(os.path.join(DATA_DIR, FEATURES_FILENAME))
        mock_read_csv.assert_any_call(os.path.join(DATA_DIR, TARGETS_FILENAME))

        mock_knn_imputer.assert_called_once()
        mock_scaler.assert_called_once()

        mock_to_csv.assert_any_call(os.path.join(DATA_DIR, CLEAN_FEATURES_FILENAME), index=False)
        mock_to_csv.assert_any_call(os.path.join(DATA_DIR, CLEAN_TARGETS_FILENAME), index=False)

        mock_joblib_dump.assert_any_call(unittest.mock.ANY, os.path.join(MODELS_DIR, IMPUTER_FILENAME))
        mock_joblib_dump.assert_any_call(unittest.mock.ANY, os.path.join(MODELS_DIR, SCALER_FILENAME))
