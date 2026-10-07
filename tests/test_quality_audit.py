import unittest
import pandas as pd
import os
from pathlib import Path
from src.data_loader import load_raw_data
from src.quality_audit import run_data_quality_audit, save_audit_report
from src.utils import get_project_root

class TestQualityAudit(unittest.TestCase):
    def setUp(self):
        self.df = load_raw_data()

    def test_run_data_quality_audit_keys(self):
        audit = run_data_quality_audit(self.df)
        self.assertIn("dataset_metrics", audit)
        self.assertIn("missing_values", audit)
        self.assertIn("whitespace_anomalies", audit)
        self.assertIn("target_analysis", audit)
        self.assertIn("feature_data_types", audit)

    def test_dataset_metrics(self):
        audit = run_data_quality_audit(self.df)
        metrics = audit["dataset_metrics"]
        self.assertEqual(metrics["total_rows"], 7111)
        self.assertEqual(metrics["total_columns"], 21)
        self.assertEqual(metrics["exact_duplicate_rows"], 20)

    def test_whitespace_anomalies(self):
        audit = run_data_quality_audit(self.df)
        ws = audit["whitespace_anomalies"]
        self.assertEqual(ws["total_charges_blanks"], 11)
        self.assertEqual(ws["tenure_zero_total_charges_blanks"], 11)

    def test_target_analysis(self):
        audit = run_data_quality_audit(self.df)
        target = audit["target_analysis"]
        self.assertEqual(target["target_column"], "Churn")
        self.assertEqual(target["counts"]["No"], 5166)
        self.assertEqual(target["counts"]["Yes"], 1873)
        self.assertEqual(target["missing_target_rows"], 72)

    def test_save_audit_report(self):
        audit = run_data_quality_audit(self.df)
        file_path = save_audit_report(audit, output_filename="test_audit.json")
        self.assertTrue(os.path.exists(file_path))
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    unittest.main()
