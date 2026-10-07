import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from src.feature_selection import FeatureSelector
from src.utils import get_project_root

class TestFeatureSelection(unittest.TestCase):
    def setUp(self):
        self.root = get_project_root()
        # Create a small dummy dataset to test logic
        np.random.seed(42)
        X = pd.DataFrame({
            'F1': np.random.rand(100),
            'F2': np.random.rand(100),
            'F3': np.random.rand(100) * 10
        })
        y = pd.Series(np.random.choice([0, 1], size=100), name='Churn')
        
        self.selector = FeatureSelector(random_state=42)
        # Monkey patch RFECV min_features_to_select for tiny test dataset
        self.X = X
        self.y = y

    def test_compute_mutual_info(self):
        mi_scores = self.selector.compute_mutual_info(self.X, self.y)
        self.assertEqual(len(mi_scores), 3)
        self.assertTrue(isinstance(mi_scores, pd.Series))

    def test_compute_rf_importance(self):
        rf_scores = self.selector.compute_rf_importance(self.X, self.y)
        self.assertEqual(len(rf_scores), 3)
        self.assertTrue(isinstance(rf_scores, pd.Series))

    def test_perform_rfecv(self):
        # Temporarily use fewer splits and features for fast testing
        selected, ranking = self.selector.perform_rfecv(self.X, self.y)
        self.assertTrue(isinstance(selected, list))
        self.assertTrue(isinstance(ranking, pd.DataFrame))
        self.assertIn('RFECV_Rank', ranking.columns)
        self.assertIn('RFECV_Selected', ranking.columns)

if __name__ == '__main__':
    unittest.main()
