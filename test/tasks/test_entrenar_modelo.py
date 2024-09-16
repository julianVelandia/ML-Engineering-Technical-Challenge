import unittest
from unittest.mock import patch
import os
import pandas as pd
from tasks.entrenar_modelo import entrenar_modelo
from config.constants import DATA_DIR, MODELS_DIR, CLEAN_FEATURES_FILENAME, CLEAN_TARGETS_FILENAME, \
    INITIAL_METRICS_FILENAME, MODEL_FILENAME
from sklearn.model_selection import train_test_split


class TestEntrenarModelo(unittest.TestCase):

    @patch('tasks.entrenar_modelo.joblib.dump')
    @patch('tasks.entrenar_modelo.open', create=True)
    @patch('tasks.entrenar_modelo.os.makedirs')
    @patch('tasks.entrenar_modelo.RandomForestClassifier.fit')
    @patch('tasks.entrenar_modelo.RandomForestClassifier.predict')
    @patch('tasks.entrenar_modelo.pd.read_csv')
    def test_entrenar_modelo(self, mock_read_csv, mock_predict, mock_fit, mock_makedirs, mock_open, mock_joblib_dump):
        mock_read_csv.side_effect = [
            pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]}),
            pd.DataFrame({'target': [0, 1, 0, 1]})
        ]

        X = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [5, 6, 7, 8]})
        y = pd.DataFrame({'target': [0, 1, 0, 1]})

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mock_predict.return_value = [0] * len(y_test)

        entrenar_modelo()

        mock_read_csv.assert_any_call(os.path.join(DATA_DIR, CLEAN_FEATURES_FILENAME))
        mock_read_csv.assert_any_call(os.path.join(DATA_DIR, CLEAN_TARGETS_FILENAME))

        mock_fit.assert_called_once()
        mock_predict.assert_called_once()

        mock_makedirs.assert_called_once_with(MODELS_DIR, exist_ok=True)
        mock_open.assert_called_once_with(os.path.join(MODELS_DIR, INITIAL_METRICS_FILENAME), 'w')
        mock_joblib_dump.assert_called_once_with(unittest.mock.ANY, os.path.join(MODELS_DIR, MODEL_FILENAME))



