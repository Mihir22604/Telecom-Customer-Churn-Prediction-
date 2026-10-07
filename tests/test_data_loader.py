import unittest
import pandas as pd
from pathlib import Path
from src.data_loader import load_raw_data, get_dataset_summary
from src.utils import get_project_root

class TestDataLoader(unittest.TestCase):
    def test_load_raw_data_shape(self):
        df = load_raw_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[0], 7111)
        self.assertEqual(df.shape[1], 21)

    def test_raw_data_columns(self):
        df = load_raw_data()
        expected_columns = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        self.assertListEqual(list(df.columns), expected_columns)

    def test_dataset_summary(self):
        df = load_raw_data()
        summary = get_dataset_summary(df)
        self.assertEqual(summary['num_rows'], 7111)
        self.assertEqual(summary['num_columns'], 21)
        self.assertEqual(summary['duplicate_rows'], 20)
        self.assertEqual(len(summary['columns_with_missing']), 17)

if __name__ == '__main__':
    unittest.main()
