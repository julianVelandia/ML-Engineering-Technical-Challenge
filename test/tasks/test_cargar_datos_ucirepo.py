import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd

from config.constants import DATA_DIR, FEATURES_FILENAME, TARGETS_FILENAME
from tasks.cargar_datos_ucirepo import cargar_datos_ucirepo


class TestCargarDatosUcirepo(unittest.TestCase):

    @patch('pandas.DataFrame.to_csv')
    @patch('tasks.cargar_datos_ucirepo.os.makedirs')
    @patch('tasks.cargar_datos_ucirepo.fetch_ucirepo')
    def test_cargar_datos(self, mock_fetch_ucirepo, mock_makedirs, mock_to_csv):
        """
        Test para verificar que cargar_datos_ucirepo descarga y guarda correctamente los datos en CSV.
        """

        mock_dataset = MagicMock()
        mock_dataset.data.features = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mock_dataset.data.targets = pd.DataFrame({'target': [0, 1]})
        mock_fetch_ucirepo.return_value = mock_dataset

        cargar_datos_ucirepo(350)

        mock_fetch_ucirepo.assert_called_once_with(id=350)

        mock_makedirs.assert_called_once_with(DATA_DIR, exist_ok=True)

        mock_to_csv.assert_any_call(os.path.join(DATA_DIR, FEATURES_FILENAME), index=False)
        mock_to_csv.assert_any_call(os.path.join(DATA_DIR, TARGETS_FILENAME), index=False)

