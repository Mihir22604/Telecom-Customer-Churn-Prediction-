import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import unittest
import joblib
import numpy as np
import pandas as pd
from src.utils import get_project_root


class TestSerialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.models_dir = cls.root / "models"
        cls.prep_path = cls.models_dir / "preprocessing_pipeline.joblib"
        cls.model_path = cls.models_dir / "best_churn_model.joblib"
        cls.pipeline_path = cls.models_dir / "full_inference_pipeline.joblib"
        cls.unified_path = cls.models_dir / "unified_churn_pipeline.joblib"
        cls.metadata_path = cls.models_dir / "model_metadata.json"

    def test_artifacts_exist(self):
        """Verify that all required Task 13 serialized artifacts exist on disk."""
        self.assertTrue(self.prep_path.exists(), f"Missing {self.prep_path}")
        self.assertTrue(self.model_path.exists(), f"Missing {self.model_path}")
        self.assertTrue(self.pipeline_path.exists(), f"Missing {self.pipeline_path}")
        self.assertTrue(self.unified_path.exists(), f"Missing {self.unified_path}")
        self.assertTrue(self.metadata_path.exists(), f"Missing {self.metadata_path}")

    def test_metadata_content(self):
        """Verify schema and content of model_metadata.json."""
        with open(self.metadata_path, "r") as f:
            meta = json.load(f)

        expected_keys = [
            "model_name", "model_version", "training_timestamp", "algorithm",
            "best_hyperparameters", "feature_count", "feature_names",
            "test_set_metrics", "artifacts"
        ]
        for key in expected_keys:
            self.assertIn(key, meta, f"Metadata missing expected key: {key}")

        self.assertEqual(meta["model_name"], "LightGBMChurnClassifier")
        self.assertEqual(meta["algorithm"], "LightGBM")
        self.assertEqual(meta["feature_count"], 28)
        self.assertEqual(len(meta["feature_names"]), 28)

        # Check required evaluation metrics exist in metadata
        metrics = meta["test_set_metrics"]
        for metric_name in ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]:
            self.assertIn(metric_name, metrics)
            self.assertGreater(metrics[metric_name], 0.0)

    def test_deserialization_and_inference(self):
        """Verify deserialization of preprocessing and model artifacts and prediction consistency."""
        prep_pipe = joblib.load(self.prep_path)
        model = joblib.load(self.model_path)
        combined = joblib.load(self.pipeline_path)

        with open(self.metadata_path, "r") as f:
            meta = json.load(f)
        feature_names = meta["feature_names"]

        # Construct synthetic sample input batch of 5 records
        np.random.seed(42)
        sample_data = {feat: np.random.rand(5) for feat in feature_names}
        # Explicitly set valid continuous features
        sample_data["tenure"] = [1.0, 12.0, 24.0, 48.0, 72.0]
        sample_data["MonthlyCharges"] = [25.0, 50.0, 70.0, 95.0, 110.0]
        sample_data["TotalCharges"] = [25.0, 600.0, 1680.0, 4560.0, 7920.0]
        sample_data["AvgMonthlyChargeRatio"] = [12.5, 46.1, 67.2, 93.0, 108.5]
        sample_data["MonthlyToTotalRatio"] = [0.96, 0.08, 0.04, 0.02, 0.01]
        sample_data["TotalServicesCount"] = [1.0, 3.0, 4.0, 6.0, 8.0]
        df_sample = pd.DataFrame(sample_data)

        # 1. Prediction using separated artifacts
        X_proc = prep_pipe.transform(df_sample)
        preds_sep = model.predict(X_proc)
        probs_sep = model.predict_proba(X_proc)[:, 1]

        # Check output dimensions
        self.assertEqual(len(preds_sep), 5)
        self.assertEqual(len(probs_sep), 5)
        self.assertTrue(np.all((probs_sep >= 0.0) & (probs_sep <= 1.0)))

        # 2. Prediction using unified pipeline dictionary
        combined_prep = combined["preprocessing_pipeline"]
        combined_model = combined["classifier"]
        X_proc_comb = combined_prep.transform(df_sample)
        preds_comb = combined_model.predict(X_proc_comb)
        probs_comb = combined_model.predict_proba(X_proc_comb)[:, 1]

        # 3. Prediction using standalone unified sklearn Pipeline
        unified_pipe = joblib.load(self.unified_path)
        preds_unified = unified_pipe.predict(df_sample)
        probs_unified = unified_pipe.predict_proba(df_sample)[:, 1]

        # 4. Verify exact consistency across all representations (up to 6 decimal places)
        np.testing.assert_array_equal(preds_sep, preds_comb)
        np.testing.assert_allclose(probs_sep, probs_comb, atol=1e-6)
        np.testing.assert_array_equal(preds_sep, preds_unified)
        np.testing.assert_allclose(probs_sep, probs_unified, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
