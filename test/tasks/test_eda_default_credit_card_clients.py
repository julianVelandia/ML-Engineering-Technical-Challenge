import os
import unittest
from unittest.mock import patch

import pandas as pd

from config.constants import DATA_DIR, FEATURES_FILENAME, REPORTS_DIR, EDA_FILE
from tasks.eda_default_credit_card_clients import generar_eda


class TestGenerarEDA(unittest.TestCase):

    @patch('tasks.eda_default_credit_card_clients.pd.read_csv')
    @patch('tasks.eda_default_credit_card_clients.os.makedirs')
    @patch('tasks.eda_default_credit_card_clients.open', create=True)
    @patch('tasks.eda_default_credit_card_clients.plt.savefig')
    def test_generar_eda(self, mock_savefig, mock_open, mock_makedirs, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({
            'LIMIT_BAL': [1000, 2000, 3000],
            'AGE': [25, 35, 45],
            'BILL_AMT1': [500, 1000, 1500],
            'PAY_AMT1': [100, 200, 300],
            'default_payment_next_month': [0, 1, 0]
        })

        generar_eda()

        mock_makedirs.assert_called_once_with(REPORTS_DIR, exist_ok=True)
        mock_open.assert_called_once_with(EDA_FILE, 'w', encoding='utf-8')

        self.assertEqual(mock_savefig.call_count, 6)

        mock_read_csv.assert_called_once_with(os.path.join(DATA_DIR, FEATURES_FILENAME))
