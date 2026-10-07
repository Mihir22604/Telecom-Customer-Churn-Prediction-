import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from src.evaluation import evaluate_models
from src.utils import get_project_root
import shutil

class TestEvaluation(unittest.TestCase):
    def setUp(self):
        self.root = get_project_root()
        # Mock dataset
        np.random.seed(42)
        self.X_test = pd.DataFrame(np.random.rand(50, 5), columns=[f'F{i}' for i in range(5)])
        self.y_test = pd.Series(np.random.choice([0, 1], size=50))
        
        self.model = LogisticRegression()
        self.model.fit(self.X_test, self.y_test)
        
        self.output_dir = self.root / "tests" / "temp_eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def test_evaluate_models(self):
        models = {'LogReg': self.model}
        metrics_df = evaluate_models(models, self.X_test, self.y_test, self.output_dir)
        
        # Check DataFrame output
        self.assertEqual(metrics_df.shape[0], 1)
        self.assertIn('F1_Score', metrics_df.columns)
        self.assertIn('ROC_AUC', metrics_df.columns)
        
        # Check file outputs
        fig_dir = self.output_dir / "reports" / "figures"
        self.assertTrue((fig_dir / "model_roc_comparison.png").exists())
        self.assertTrue((fig_dir / "model_pr_comparison.png").exists())
        self.assertTrue((fig_dir / "confusion_matrices.png").exists())

    def tearDown(self):
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

if __name__ == '__main__':
    unittest.main()
