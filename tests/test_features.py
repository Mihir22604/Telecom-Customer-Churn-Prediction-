import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from src.features import FeatureEngineeringTransformer, engineer_features
from src.data_loader import load_raw_data
from src.cleaning import TelecomDataCleaner
from src.utils import get_project_root


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.project_root = get_project_root()
        self.cleaned_csv = self.project_root / "data" / "processed" / "cleaned_churn_data.csv"
        self.df_clean = pd.read_csv(self.cleaned_csv)
        self.transformer = FeatureEngineeringTransformer(scale_numerical=True)
        self.df_eng = self.transformer.fit_transform(self.df_clean)

    def test_output_shape_rows(self):
        # Row count must remain 7019 after feature engineering
        self.assertEqual(self.df_eng.shape[0], 7019)

    def test_output_feature_count(self):
        # 38 predictor features + customerID + Churn = 40 columns total
        self.assertEqual(self.df_eng.shape[1], 40)

    def test_zero_nulls(self):
        self.assertEqual(self.df_eng.isnull().sum().sum(), 0)

    def test_domain_features_exist(self):
        for col in ['AvgMonthlyChargeRatio', 'MonthlyToTotalRatio',
                    'TotalServicesCount', 'HasTechSupportSecurity', 'IsAutomaticPayment']:
            self.assertIn(col, self.df_eng.columns, f"Missing engineered feature: {col}")

    def test_tenure_cohort_ohe_exists(self):
        for col in ['TenureCohort_12-24m', 'TenureCohort_24-48m', 'TenureCohort_48-72m']:
            self.assertIn(col, self.df_eng.columns, f"Missing tenure cohort OHE: {col}")

    def test_binary_encoding_range(self):
        for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            if col in self.df_eng.columns:
                unique_vals = set(self.df_eng[col].unique())
                self.assertTrue(unique_vals.issubset({0, 1}),
                                f"Column '{col}' has non-binary values: {unique_vals}")

    def test_churn_binary_encoding(self):
        self.assertIn('Churn', self.df_eng.columns)
        unique_targets = set(self.df_eng['Churn'].unique())
        self.assertEqual(unique_targets, {0, 1})

    def test_saved_engineered_file_exists(self):
        output_path = self.project_root / "data" / "processed" / "engineered_churn_data.csv"
        self.assertTrue(output_path.exists())
        df_saved = pd.read_csv(output_path)
        self.assertEqual(df_saved.shape, (7019, 40))


if __name__ == '__main__':
    unittest.main()
