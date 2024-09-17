import os
import unittest
from unittest.mock import patch

from config.constants import MODELS_DIR, RETRAIN_FLAG_FILENAME
from tasks.reentrenar_modelo import reentrenar_modelo


class TestReentrenarModelo(unittest.TestCase):

    @patch('tasks.reentrenar_modelo.ajustar_hiperparametros')
    @patch('tasks.reentrenar_modelo.os.remove')
    @patch('tasks.reentrenar_modelo.os.path.exists')
    def test_reentrenar_modelo_con_retrain_flag(self, mock_exists, mock_remove, mock_ajustar_hiperparametros):
        mock_exists.return_value = True

        reentrenar_modelo()

        retrain_flag_path = os.path.join(MODELS_DIR, RETRAIN_FLAG_FILENAME)
        mock_remove.assert_called_once_with(retrain_flag_path)
        mock_ajustar_hiperparametros.assert_called_once()

    @patch('tasks.reentrenar_modelo.ajustar_hiperparametros')
    @patch('tasks.reentrenar_modelo.os.remove')
    @patch('tasks.reentrenar_modelo.os.path.exists')
    def test_reentrenar_modelo_sin_retrain_flag(self, mock_exists, mock_remove, mock_ajustar_hiperparametros):
        mock_exists.return_value = False

        reentrenar_modelo()

        mock_remove.assert_not_called()
        mock_ajustar_hiperparametros.assert_not_called()
