import unittest
import pandas as pd
import os
from pathlib import Path
from src.eda import generate_eda_figures, compute_eda_summary_statistics
from src.utils import get_project_root

class TestEDA(unittest.TestCase):
    def setUp(self):
        self.project_root = get_project_root()
        self.cleaned_csv = self.project_root / "data" / "processed" / "cleaned_churn_data.csv"
        self.df = pd.read_csv(self.cleaned_csv)

    def test_generate_eda_figures(self):
        figures_dir = self.project_root / "reports" / "figures"
        figures = generate_eda_figures(processed_df_path=self.cleaned_csv, figures_dir=figures_dir)
        
        self.assertIn("target_distribution", figures)
        self.assertIn("numerical_distributions", figures)
        self.assertIn("categorical_vs_churn", figures)
        self.assertIn("multivariate_tenure_charges", figures)
        
        for name, fig_path in figures.items():
            self.assertTrue(os.path.exists(fig_path), f"Figure {name} does not exist at {fig_path}")

    def test_compute_eda_summary_statistics(self):
        summary = compute_eda_summary_statistics(self.df)
        self.assertIn("numerical_by_churn", summary)
        self.assertIn("churn_rate_by_contract", summary)
        self.assertIn("churn_rate_by_internet_service", summary)
        
        # Check specific values
        num_stats = summary["numerical_by_churn"]
        self.assertIn("tenure", num_stats)
        self.assertIn("MonthlyCharges", num_stats)

if __name__ == '__main__':
    unittest.main()
