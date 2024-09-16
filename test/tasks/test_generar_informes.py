import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from tasks.generar_informes import generar_informes
from config.constants import MODELS_DIR, DATA_DIR, REPORTS_DIR, CLEAN_FEATURES_FILENAME, CLEAN_TARGETS_FILENAME, \
    BEST_MODEL_FILENAME

class TestGenerarInformes(unittest.TestCase):

    @patch('tasks.generar_informes.plt.savefig')
    @patch('tasks.generar_informes.pd.read_csv')
    @patch('tasks.generar_informes.os.makedirs')
    @patch('tasks.generar_informes.joblib.load')
    def test_generar_informes(self, mock_load, mock_makedirs, mock_read_csv, mock_savefig):
        mock_read_csv.side_effect = [
            pd.DataFrame({
                'col1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'col2': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            }),
            pd.DataFrame({'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})  # Targets con más filas
        ]

        # Simulación del modelo y predicciones
        mock_best_model = MagicMock()
        mock_best_model.predict.return_value = [0, 1]
        mock_best_model.predict_proba.return_value = [[0.6, 0.4], [0.2, 0.8]]
        mock_load.return_value = mock_best_model

        generar_informes()

        mock_load.assert_called_once_with(os.path.join(MODELS_DIR, BEST_MODEL_FILENAME))
        mock_read_csv.assert_any_call(os.path.join(DATA_DIR, CLEAN_FEATURES_FILENAME))
        mock_read_csv.assert_any_call(os.path.join(DATA_DIR, CLEAN_TARGETS_FILENAME))
        mock_makedirs.assert_called_once_with(REPORTS_DIR, exist_ok=True)
        mock_savefig.assert_any_call(os.path.join(REPORTS_DIR, 'confusion_matrix.png'))
        mock_savefig.assert_any_call(os.path.join(REPORTS_DIR, 'roc_curve.png'))
