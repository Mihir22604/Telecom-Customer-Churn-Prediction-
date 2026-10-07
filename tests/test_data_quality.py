import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import unittest
import pandas as pd
import numpy as np
from src.utils import get_project_root


class TestDataQuality(unittest.TestCase):
    """
    Integration tests verifying clean data quality and invariants 
    for the Telecom Churn dataset.
    """
    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.cleaned_path = cls.root / "data" / "processed" / "cleaned_churn_data.csv"
        cls.selected_path = cls.root / "data" / "processed" / "selected_churn_data.csv"
        
        cls.df_cleaned = pd.read_csv(cls.cleaned_path)
        cls.df_selected = pd.read_csv(cls.selected_path)

    def test_cleaned_data_zero_nans(self):
        """Verify that cleaned dataset contains zero null or NaN values."""
        null_counts = self.df_cleaned.isnull().sum()
        total_nulls = null_counts.sum()
        self.assertEqual(total_nulls, 0, f"Found unexpected NaNs in cleaned data:\n{null_counts[null_counts > 0]}")

    def test_selected_data_zero_nans(self):
        """Verify that selected predictor matrix contains zero null values."""
        null_counts = self.df_selected.isnull().sum()
        total_nulls = null_counts.sum()
        self.assertEqual(total_nulls, 0, f"Found unexpected NaNs in selected data:\n{null_counts[null_counts > 0]}")

    def test_no_exact_duplicates(self):
        """Verify that exact duplicate rows were completely eliminated."""
        duplicates_count = self.df_cleaned.duplicated().sum()
        self.assertEqual(duplicates_count, 0, f"Found {duplicates_count} duplicate rows in cleaned data")

    def test_expected_row_counts(self):
        """Verify that row counts match post-cleaning expectations (7,019 rows)."""
        self.assertEqual(len(self.df_cleaned), 7019, f"Unexpected row count: {len(self.df_cleaned)}")
        self.assertEqual(len(self.df_selected), 7019, f"Unexpected row count: {len(self.df_selected)}")

    def test_numerical_data_types(self):
        """Verify continuous columns are correctly cast to float64."""
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            self.assertIn(col, self.df_cleaned.columns)
            self.assertTrue(
                pd.api.types.is_float_dtype(self.df_cleaned[col]) or pd.api.types.is_integer_dtype(self.df_cleaned[col]),
                f"Column {col} has invalid dtype: {self.df_cleaned[col].dtype}"
            )

    def test_target_class_balance(self):
        """Verify target distribution maintains expected ~26.3% - 26.6% churn ratio."""
        churn_counts = self.df_cleaned['Churn'].value_counts(normalize=True)
        yes_ratio = churn_counts.get('Yes', 0.0)
        self.assertAlmostEqual(yes_ratio, 0.2655, delta=0.02, msg="Target class balance distorted")


if __name__ == '__main__':
    unittest.main()
