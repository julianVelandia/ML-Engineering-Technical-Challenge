import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

from config.constants import MODELS_DIR, BEST_MODEL_FILENAME
from tasks.ajustar_hiperparametros import ajustar_hiperparametros


class TestAjustarHiperparametros(unittest.TestCase):

    @patch('tasks.ajustar_hiperparametros.joblib.dump')
    @patch('tasks.ajustar_hiperparametros.open', create=True)
    @patch('tasks.ajustar_hiperparametros.GridSearchCV')
    @patch('tasks.ajustar_hiperparametros.train_test_split')
    @patch('tasks.ajustar_hiperparametros.pd.read_csv')
    def test_ajustar_hiperparametros(self, mock_read_csv, mock_train_test_split, mock_grid_search_cv, mock_open,
                                     mock_joblib_dump):
        X = pd.DataFrame({'col1': range(10), 'col2': range(10, 20)})
        y = pd.DataFrame({'target': [0, 1] * 5})

        mock_read_csv.side_effect = [X, y]

        X_train = X.iloc[:8]
        X_test = X.iloc[8:]
        y_train = y.iloc[:8]
        y_test = y.iloc[8:]

        mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)

        mock_grid_search_instance = MagicMock()
        mock_grid_search_instance.best_estimator_ = MagicMock()
        mock_grid_search_instance.best_estimator_.predict.return_value = [0, 1]
        mock_grid_search_instance.best_params_ = {'n_estimators': 100, 'max_depth': 10}
        mock_grid_search_cv.return_value = mock_grid_search_instance

        ajustar_hiperparametros()

        mock_train_test_split.assert_called_once()
        mock_grid_search_instance.fit.assert_called_once()
        mock_joblib_dump.assert_called_once_with(
            mock_grid_search_instance.best_estimator_,
            os.path.join(MODELS_DIR, BEST_MODEL_FILENAME)
        )
