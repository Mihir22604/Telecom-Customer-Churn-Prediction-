import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_raw_data
from src.cleaning import TelecomDataCleaner, clean_telecom_data
from src.utils import get_project_root

class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        self.raw_df = load_raw_data()
        self.cleaner = TelecomDataCleaner()
        self.cleaned_df = self.cleaner.fit_transform(self.raw_df)

    def test_cleaned_dataset_shape(self):
        # 7111 raw rows - 72 missing target - 20 exact duplicates = 7019 rows
        self.assertEqual(self.cleaned_df.shape[0], 7019)
        self.assertEqual(self.cleaned_df.shape[1], 21)

    def test_zero_missing_values(self):
        null_count = self.cleaned_df.isnull().sum().sum()
        self.assertEqual(null_count, 0)

    def test_total_charges_numeric_coercion(self):
        self.assertTrue(pd.api.types.is_numeric_dtype(self.cleaned_df['TotalCharges']))
        self.assertEqual(self.cleaned_df['TotalCharges'].dtype, np.float64)

    def test_zero_exact_duplicates(self):
        duplicate_count = int(self.cleaned_df.duplicated().sum())
        self.assertEqual(duplicate_count, 0)

    def test_target_churn_no_nans(self):
        self.assertEqual(self.cleaned_df['Churn'].isnull().sum(), 0)
        unique_targets = set(self.cleaned_df['Churn'].unique())
        self.assertEqual(unique_targets, {'Yes', 'No'})

    def test_saved_cleaned_file_exists(self):
        project_root = get_project_root()
        processed_file = project_root / "data" / "processed" / "cleaned_churn_data.csv"
        self.assertTrue(processed_file.exists())
        
        df_saved = pd.read_csv(processed_file)
        self.assertEqual(df_saved.shape[0], 7019)
        self.assertEqual(df_saved.isnull().sum().sum(), 0)

if __name__ == '__main__':
    unittest.main()
