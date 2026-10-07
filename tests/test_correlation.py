import unittest
import pandas as pd
import os
from pathlib import Path
from src.correlation import run_correlation_analysis, compute_vif
from src.utils import get_project_root

class TestCorrelation(unittest.TestCase):
    def setUp(self):
        self.project_root = get_project_root()
        self.cleaned_csv = self.project_root / "data" / "processed" / "cleaned_churn_data.csv"
        self.df = pd.read_csv(self.cleaned_csv)

    def test_compute_vif(self):
        vif_df = compute_vif(self.df, ['tenure', 'MonthlyCharges', 'TotalCharges'])
        self.assertEqual(len(vif_df), 3)
        self.assertIn("Feature", vif_df.columns)
        self.assertIn("VIF", vif_df.columns)
        
        # Verify TotalCharges has high VIF > 8.0
        total_charges_vif = float(vif_df[vif_df['Feature'] == 'TotalCharges']['VIF'].values[0])
        self.assertGreater(total_charges_vif, 8.0)

    def test_run_correlation_analysis(self):
        summary = run_correlation_analysis(processed_df_path=self.cleaned_csv)
        self.assertIn("pearson_correlation", summary)
        self.assertIn("spearman_correlation", summary)
        self.assertIn("vif_scores", summary)
        self.assertIn("target_correlations_with_churn", summary)
        
        # Check heatmap figure generation
        heatmap_path = self.project_root / "reports" / "figures" / "05_correlation_heatmap.png"
        self.assertTrue(os.path.exists(heatmap_path))
        
        # Check JSON output export
        json_path = self.project_root / "reports" / "correlation_analysis.json"
        self.assertTrue(os.path.exists(json_path))

if __name__ == '__main__':
    unittest.main()
